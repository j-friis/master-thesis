class ConvNetRGB(nn.Module):
    def __init__(self):
        super(ConvNetRGB, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ).cuda()
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)).cuda()
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.ReLU()).cuda()
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
            nn.ReLU()).cuda()

        self.down1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1).cuda()
        
        self.down2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                        kernel_size=3, stride=2, padding=1, output_padding=1).cuda()
        
        self.down3 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                        kernel_size=3, stride=2, padding=1, output_padding=1).cuda()
        
        self.down4 = nn.ConvTranspose2d(in_channels=32, out_channels=1,
                                        kernel_size=3, stride=1, padding=1).cuda()
        

        
    def forward(self, x):
        out = self.layer1(x)
        # print(f"self.layer1 {out.shape}")
        out = self.layer2(out)
        # print(f"self.layer2 {out.shape}")
        out = self.layer3(out)
        # print(f"self.layer3 {out.shape}")
        out = self.layer4(out)
        # print(f"self.layer4 {out.shape}")
        out = self.down1(out)
        # print(f"self.down1 {out.shape}")
        out = self.down2(out)
        # print(f"self.down2 {out.shape}")
        out = self.down3(out)
        # print(f"self.down3 {out.shape}")
        out = self.down4(out)
        # print(f"self.down4 {out.shape}")
        
        #out = torch.sigmoid(out)
        return out
    

class Doub_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1).cuda()
        self.relu  = nn.ReLU().cuda()
        self.batchnor = nn.BatchNorm2d(out_ch, affine=False).cuda()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1).cuda()
    
    def forward(self, x):
        return self.relu(self.batchnor(self.conv2(self.relu(self.batchnor(self.conv1(x))))))


class Down(nn.Module):
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        self.Doub_convs = nn.ModuleList([Doub_conv(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for Doub_conv in self.Doub_convs:
            x = Doub_conv(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Up(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.Doub_convs = nn.ModuleList([Doub_conv(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.Doub_convs[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs).cuda()
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True, out_sz=(584,565)):
        super().__init__()
        self.encoder     = Down(enc_chs)
        self.decoder     = Up(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out      = torch.sigmoid(out)
        if self.retain_dim:
            out = F.interpolate(out, (x.shape[2],x.shape[3]))
        return out

bestModel, ValidationLossArray, TrainingLossArray = ConvNetTraining(trainloader, valloader, UNet().cuda(), nn.MSELoss(), 0.001, 1000)