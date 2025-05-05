import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn.functional as F
from torch.optim import Adam


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + skip


class ConditionLayer(nn.Module):
    def __init__(self, condition_dim, channels, spatial_dims=None):
        super(ConditionLayer, self).__init__()
        self.hidden_dim = 128
        self.condition_encoder = nn.Linear(condition_dim, self.hidden_dim)
        self.gamma_layer = nn.Linear(self.hidden_dim, channels)
        self.beta_layer = nn.Linear(self.hidden_dim, channels)
        self.spatial_dims = spatial_dims
        
    def forward(self, x, condition):
        batch_size = x.shape[0]
        condition_features = self.condition_encoder(condition)
        gamma = self.gamma_layer(condition_features)
        beta = self.beta_layer(condition_features)
        
        if len(x.shape) == 4:  
            gamma = gamma.view(batch_size, x.shape[1], 1, 1)
            beta = beta.view(batch_size, x.shape[1], 1, 1)
        
        return (1 + gamma) * x + beta


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        attn_output, _ = self.attention(query, key_value, key_value)
        x = query + attn_output
        x = self.norm(x)
        
        return x

class AdvancedLatentPredictor(nn.Module):
    """A latent space predictor based on attention mechanism"""
    def __init__(self, condition_rep_dim, condition_nk_dim, latent_dim, 
                 embed_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super(AdvancedLatentPredictor, self).__init__()
        
        self.condition_rep_dim = condition_rep_dim
        self.condition_nk_dim = condition_nk_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        
        self.embed_rep = nn.Sequential(
            nn.Linear(condition_rep_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.embed_nk = nn.Sequential(
            nn.Linear(condition_nk_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # self-attention layers
        self.self_attention_rep = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.self_attention_nk = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # cross-attention layers
        self.cross_attention_rep_to_nk = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attention_nk_to_rep = CrossAttentionBlock(embed_dim, num_heads, dropout)
        
        # fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # output layer
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, latent_dim)
        )
        
    def forward(self, c_rep, c_nk):
        batch_size = c_rep.shape[0]
        
        # embed condition variables
        rep_embed = self.embed_rep(c_rep).unsqueeze(1)  # [batch_size, 1, embed_dim]
        nk_embed = self.embed_nk(c_nk).unsqueeze(1)     # [batch_size, 1, embed_dim]
        
        # apply self-attention
        for attn_layer in self.self_attention_rep:
            rep_embed = attn_layer(rep_embed)
            
        for attn_layer in self.self_attention_nk:
            nk_embed = attn_layer(nk_embed)
        
        # apply cross-attention
        rep_cross = self.cross_attention_rep_to_nk(rep_embed, nk_embed)
        nk_cross = self.cross_attention_nk_to_rep(nk_embed, rep_embed)
        
        # fusion representations
        combined = torch.cat([rep_cross.squeeze(1), nk_cross.squeeze(1)], dim=1)
        fused = self.fusion_layer(combined)
        
        # generate latent vector
        latent_pred = self.output_layers(fused)
        
        return latent_pred

class JCVAE(nn.Module):
    """Joint Conditional Variational Autoencoder with Advanced Latent Predictor"""
    def __init__(self, latent_dim=12, condition_rep_dim=40, condition_nk_dim=4):
        """
        1. Encoder:
           - Takes 40-channel input images through convolutional layers with residual connections
           - Uses condition variables (rep and nk) via ConditionLayer to modulate features
           - Downsamples with max pooling and processes through fully connected layers
           - Outputs mean (mu) and log variance (logvar) for latent space sampling
        
        2. Decoder:
           - Takes sampled latent vector concatenated with condition variables
           - Processes through fully connected layers with condition modulation
           - Upsamples using transposed convolutions with residual connections
           - Reconstructs 40-channel output image
           
        3. Latent Predictor:
           - Separate network that directly predicts latent vectors from condition variables
           - Uses self-attention and cross-attention between rep and nk embeddings
           - Fuses information through multiple attention layers
           - Enables direct latent space sampling without encoding images
        
        Key Features:
        - Residual connections for better gradient flow
        - Condition modulation at multiple levels
        - Attention mechanisms for condition processing
        - Batch normalization for training stability
        - Multiple loss components (reconstruction, KL divergence, prediction)
        
        Input Dimensions:
        - Image: [batch_size, 40, 28, 28]
        - Rep: [batch_size, condition_rep_dim]
        - NK: [batch_size, condition_nk_dim]
        
        Latent Space: {latent_dim} dimensions
        """
        super(JCVAE, self).__init__()
        
        # save dimensions of condition variables and latent space
        self.latent_dim = latent_dim
        self.condition_rep_dim = condition_rep_dim
        self.condition_nk_dim = condition_nk_dim
        self.condition_dim = condition_rep_dim + condition_nk_dim
        
        # encoder network with residual connections
        self.encoder_conv1 = nn.Conv2d(40, 60, kernel_size=3, padding=1)
        self.encoder_res1 = ResidualBlock(60)
        self.encoder_pool1 = nn.MaxPool2d(2, 2)
        self.encoder_conv2 = nn.Conv2d(60, 32, kernel_size=3, padding=1)
        self.encoder_res2 = ResidualBlock(32)
        self.encoder_pool2 = nn.MaxPool2d(2, 2)
        
        # condition application layers
        self.condition_encoder1 = ConditionLayer(self.condition_dim, 60)
        self.condition_encoder2 = ConditionLayer(self.condition_dim, 32)
        
        # flattened feature size
        self.flattened_size = 32 * 7 * 7  
        
        # encoder's fully connected layer
        self.encoder_fc1 = nn.Linear(self.flattened_size, 392*2)
        self.encoder_fc2 = nn.Linear(392*2, latent_dim*2)
        
        # decoder's fully connected layer
        self.decoder_fc1 = nn.Linear(latent_dim + self.condition_dim, 392*2)
        self.decoder_fc2 = nn.Linear(392*2, 392*4)
        
        # condition application layers - decoder
        self.condition_decoder1 = ConditionLayer(self.condition_dim, 392*2)
        self.condition_decoder2 = ConditionLayer(self.condition_dim, 392*4)
        
        # decoder's transposed convolutional layers
        self.decoder_unflatten = nn.Unflatten(1, (32, 7, 7))
        self.decoder_tconv1 = nn.ConvTranspose2d(32, 60, kernel_size=3, stride=2)
        self.decoder_res1 = ResidualBlock(60)
        self.decoder_tconv2 = nn.ConvTranspose2d(60, 40, kernel_size=2, stride=2)
        self.decoder_res2 = ResidualBlock(40)
        
        # decoder's condition layers
        self.condition_decoder_conv1 = ConditionLayer(self.condition_dim, 60)
        self.condition_decoder_conv2 = ConditionLayer(self.condition_dim, 40)
        
        # batch normalization layers
        self.bn1 = nn.BatchNorm2d(60)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(60)
        self.bn4 = nn.BatchNorm2d(40)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # advanced latent predictor - using attention mechanism
        self.latent_predictor = AdvancedLatentPredictor(
            condition_rep_dim=condition_rep_dim,
            condition_nk_dim=condition_nk_dim,
            latent_dim=latent_dim
        )

    def encode(self, x, c_rep, c_nk):

        # merge conditions
        c = torch.cat([c_rep, c_nk], dim=1)
        
        x = self.encoder_conv1(x)
        x = F.relu(self.bn1(x))
        x = self.condition_encoder1(x, c)  # apply conditions
        x = self.encoder_res1(x)  # residual connection
        x = self.encoder_pool1(x)
        
        x = self.encoder_conv2(x)
        x = F.relu(self.bn2(x))
        x = self.condition_encoder2(x, c)  # apply conditions
        x = self.encoder_res2(x)  # residual connection
        x = self.encoder_pool2(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fully connected layer
        x = F.relu(self.encoder_fc1(x))
        x = self.dropout(x)
        x = self.encoder_fc2(x)
        
        # separate mean and log variance
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c_rep, c_nk):
        # merge latent vector and conditions
        c = torch.cat([c_rep, c_nk], dim=1)
        z_cond = torch.cat([z, c], dim=1)
        
        # pass through decoder's fully connected layer, with conditions
        x = F.relu(self.decoder_fc1(z_cond))
        x = self.condition_decoder1(x, c)  # apply conditions
        x = self.dropout(x)
        
        x = F.relu(self.decoder_fc2(x))
        x = self.condition_decoder2(x, c)  # apply conditions
        
        # restore 2D shape
        x = self.decoder_unflatten(x)
        
        # transpose convolution, with conditions and residual
        x = self.decoder_tconv1(x)
        x = F.relu(self.bn3(x))
        x = self.condition_decoder_conv1(x, c)  # apply conditions
        x = self.decoder_res1(x)  # residual connection
        
        x = self.decoder_tconv2(x)
        x = F.relu(self.bn4(x))
        x = self.condition_decoder_conv2(x, c)  # apply conditions
        x = self.decoder_res2(x)  # residual connection
        
        x = torch.sigmoid(x)  # ensure output in [0,1] range
        return x
    
    def predict_latent(self, c_rep, c_nk):
        # predict latent vector from rep+nk
        return self.latent_predictor(c_rep, c_nk)
    
    def forward(self, x, c_rep, c_nk):
        # CVAE part: encode, reparameterize, decode
        mu, logvar = self.encode(x, c_rep, c_nk)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c_rep, c_nk)
        
        # predictor part: predict latent vector
        z_pred = self.predict_latent(c_rep, c_nk)
        
        # decode from predicted latent vector
        recon_x_pred = self.decode(z_pred, c_rep, c_nk)
        
        return recon_x, mu, logvar, z_pred, recon_x_pred
    
    def get_latent_space(self, x, c_rep, c_nk):
        mu, logvar = self.encode(x, c_rep, c_nk)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        return self
    
    def generate_from_conditions(self, c_rep, c_nk): 
        # predict latent vector
        z_pred = self.predict_latent(c_rep, c_nk)
        
        # decode to generate wavefunction
        return self.decode(z_pred, c_rep, c_nk)


# extended loss function: includes CVAE loss and predictor loss
def joint_loss_function(
    recon_x, x, mu, logvar,                  # CVAE part
    z_pred, mu_target,                       # predictor part
    recon_x_pred,                            # predictor+decoder reconstruction
    beta=1.0, predictor_weight=1.0, direct_recon_weight=1.0
):      
    # CVAE reconstruction loss
    reconstruction_function = nn.MSELoss(reduction='sum')
    BCE = reconstruction_function(recon_x, x)
     
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # predictor loss - difference between predicted latent vector and true latent vector
    predictor_loss = nn.MSELoss(reduction='sum')(z_pred, mu_target.detach())
    
    # direct reconstruction loss - difference between reconstructed wavefunction from predicted latent vector and true wavefunction
    direct_recon_loss = reconstruction_function(recon_x_pred, x)
    
    # total loss
    total_loss = BCE + beta * KLD + predictor_weight * predictor_loss + direct_recon_weight * direct_recon_loss
    
    return total_loss, BCE, KLD, predictor_loss, direct_recon_loss


if __name__ == "__main__":
    import h5py
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    # load data
    file_path = "wfs_merged.h5"

    with h5py.File(file_path, "r") as file:
        data = file["data"][:]
        energy = file['energy'][:]
        information = file['information'][:]
        nk_info = file['nk_info'][:]  # load k-point information
        rep = file['rep'][:]  # load representation/descriptor
    
    # find states near Fermi Level
    energy_index = list(np.where((energy < 1) & (-1 < energy))[0])
    energy = energy[energy_index]
    data = data[energy_index]
    information = information[energy_index]
    nk_info = nk_info[energy_index]  # same filtering condition variables
    rep = rep[energy_index]  # same filtering condition variables
    information[...,-1] = energy
    
    # data preprocessing - normalize condition variables
    nk_info = (nk_info - nk_info.mean(0)) / (nk_info.std(0) + 1e-8)
    rep = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    
    # normalize data
    max_image = np.max(data, axis=(1, 2, 3))
    data = data / max_image[:, np.newaxis, np.newaxis, np.newaxis]
    
    # convert to PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)
    information = torch.tensor(information, dtype=torch.float32)
    nk_info = torch.tensor(nk_info, dtype=torch.float32)
    rep = torch.tensor(rep, dtype=torch.float32)
    
    # check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # split training set and test set - keep index consistency
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_rep = rep[train_indices]
    test_rep = rep[test_indices]
    train_nk = nk_info[train_indices]
    test_nk = nk_info[test_indices]
    
    print('Train data size:', train_data.shape)
    print('Test data size:', test_data.shape)
    
    # create batch DataLoader
    batch_size = 128
    train_data_loader = DataLoader(
        TensorDataset(train_data, train_rep, train_nk), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_data_loader = DataLoader(
        TensorDataset(test_data, test_rep, test_nk), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # initialize JCVAE model
    epochs = 300  # increase epochs
    initial_beta = 0.01
    max_beta = 0.3
    beta_warmup_epochs = 50
    
    # predictor weight - control loss weight of direct prediction of latent vector
    initial_predictor_weight = 0.2  # slightly increase initial weight
    max_predictor_weight = 1.5      # increase max weight
    predictor_warmup_epochs = 30
    
    # direct reconstruction weight - control loss weight of direct reconstruction of wavefunction from predicted latent vector
    initial_direct_recon_weight = 0.5
    max_direct_recon_weight = 2.5   # increase max weight
    direct_recon_warmup_epochs = 40
    
    # create advanced JCVAE model
    model = JCVAE(latent_dim=12, condition_rep_dim=rep.shape[1], condition_nk_dim=nk_info.shape[1]).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    model.train()
    train_error_cvae = np.zeros(epochs)
    train_error_pred = np.zeros(epochs)
    test_error_cvae = np.zeros(epochs)
    test_error_pred = np.zeros(epochs)
    variance_data = np.var(data.numpy())
    
    criterion_image = nn.MSELoss()
    criterion_image = criterion_image.to(device)
    
    for epoch in range(epochs):

        beta = min(initial_beta + (max_beta - initial_beta) * epoch / beta_warmup_epochs, max_beta)
        predictor_weight = min(initial_predictor_weight + (max_predictor_weight - initial_predictor_weight) * epoch / predictor_warmup_epochs, max_predictor_weight)
        direct_recon_weight = min(initial_direct_recon_weight + (max_direct_recon_weight - initial_direct_recon_weight) * epoch / direct_recon_warmup_epochs, max_direct_recon_weight)
        
        train_loss = 0
        train_loss_cvae = 0
        train_loss_kld = 0
        train_loss_pred = 0
        train_loss_direct = 0
        
        model.train()
        for batch_data in train_data_loader:
            # unpack data and condition variables
            data_batch, rep_batch, nk_batch = batch_data
            data_batch = data_batch.to(device)
            rep_batch = rep_batch.to(device)
            nk_batch = nk_batch.to(device)
            
            optimizer.zero_grad()
                
            recon_batch, mu, logvar, z_pred, recon_pred = model(data_batch, rep_batch, nk_batch)
            

            loss, bce, kld, pred_loss, direct_loss = joint_loss_function(
            recon_batch, data_batch, mu, logvar,
            z_pred, mu,
            recon_pred,               
            beta=beta, 
            predictor_weight=predictor_weight,
            direct_recon_weight=direct_recon_weight
            )
            
            loss.backward()
            train_loss += loss.item()
            train_loss_cvae += bce.item()
            train_loss_kld += kld.item()
            train_loss_pred += pred_loss.item()
            train_loss_direct += direct_loss.item()
            
            optimizer.step()
        

        loss_image_cvae = criterion_image(recon_batch, data_batch)
        loss_image_pred = criterion_image(recon_pred, data_batch)
        
        train_error_cvae[epoch] = loss_image_cvae.item()
        train_error_pred[epoch] = loss_image_pred.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Beta: {beta:.2f}, PW: {predictor_weight:.2f}, DRW: {direct_recon_weight:.2f}")
        print(f"Train Loss: {train_loss/len(train_data_loader):.4f}, CVAE: {train_loss_cvae/len(train_data_loader):.4f}, KLD: {train_loss_kld/len(train_data_loader):.4f}")
        print(f"Pred Loss: {train_loss_pred/len(train_data_loader):.4f}, Direct: {train_loss_direct/len(train_data_loader):.4f}")
        print(f"R^2(train-CVAE): {1 - (loss_image_cvae.item() / variance_data):.4f}, R^2(train-Pred): {1 - (loss_image_pred.item() / variance_data):.4f}")
        
        # evaluate
        model.eval()
        test_loss = 0.0
        test_loss_cvae = 0.0
        test_loss_pred = 0.0
        
        with torch.no_grad():
            for batch_data in test_data_loader:
                data_batch, rep_batch, nk_batch = batch_data
                data_batch = data_batch.to(device)
                rep_batch = rep_batch.to(device)
                nk_batch = nk_batch.to(device)
                
                # forward propagation
                recon_batch, mu, logvar, z_pred, recon_pred = model(data_batch, rep_batch, nk_batch)
                
                # calculate loss
                loss_image_cvae = criterion_image(recon_batch, data_batch)
                loss_image_pred = criterion_image(recon_pred, data_batch)
                
                test_loss_cvae += loss_image_cvae.item()
                test_loss_pred += loss_image_pred.item()
            
            # calculate average test loss
            avg_test_loss_cvae = test_loss_cvae / len(test_data_loader)
            avg_test_loss_pred = test_loss_pred / len(test_data_loader)
            
            # update learning rate scheduler - use predictor loss
            scheduler.step(avg_test_loss_pred)
        
        # save test set R^2
        test_error_cvae[epoch] = avg_test_loss_cvae
        test_error_pred[epoch] = avg_test_loss_pred
        
        print(f"R^2(test-CVAE): {1 - (avg_test_loss_cvae / variance_data):.4f}, R^2(test-Pred): {1 - (avg_test_loss_pred / variance_data):.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # early stop condition - based on predictor performance
        break_out_condition_r2 = 0.85
        if (1 - (avg_test_loss_pred / variance_data) > break_out_condition_r2):
            print(f'Predictor R^2 > {break_out_condition_r2}, training completed!')
            break
    
    print("Training completed.")
    torch.save(model.state_dict(), "JCVAE_model.pth")
    
    # save latent space and condition data
    print("Saving latent space representation...")
    model.eval()
    with torch.no_grad():
        all_data = data.to(device)
        all_rep = rep.to(device)
        all_nk = nk_info.to(device)
        
        # get latent space representation - CVAE
        latent_z, mu, logvar = model.get_latent_space(all_data, all_rep, all_nk)
        
        # get predicted latent space representation - predictor
        pred_z = model.predict_latent(all_rep, all_nk)
        
        # generate wavefunction - from predicted latent space
        gen_data = model.generate_from_conditions(all_rep, all_nk)
        
        # move back to CPU
        latent_z = latent_z.cpu().numpy()
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
        pred_z = pred_z.cpu().numpy()
        gen_data = gen_data.cpu().numpy()
        all_rep = all_rep.cpu().numpy()
        all_nk = all_nk.cpu().numpy()
        
        # save as h5 file
        with h5py.File('latent_space_data.h5', 'w') as f:
            f.create_dataset('latent_z', data=latent_z)
            f.create_dataset('mu', data=mu)
            f.create_dataset('logvar', data=logvar)
            f.create_dataset('pred_z', data=pred_z)
            f.create_dataset('gen_data', data=gen_data)
            f.create_dataset('rep', data=all_rep)
            f.create_dataset('nk_info', data=all_nk)
            f.create_dataset('energy', data=energy)
            
        # save training history
        np.savez('training_history.npz',
                train_error_cvae=train_error_cvae,
                train_error_pred=train_error_pred,
                test_error_cvae=test_error_cvae,
                test_error_pred=test_error_pred)
