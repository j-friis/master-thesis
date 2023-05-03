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