import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF 
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary



class ResidualBlock(nn.Module):  
    def __init__(self, input_channels, output_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.mish = nn.Mish()
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1)  
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride=stride, padding = 1, bias = True)
        self.dropout = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, bias = True)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride=stride, bias = True)
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.mish(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.mish(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn3(out)
        out = self.mish(out)
        out = self.conv3(out)
        residual = self.conv4(x)
        out += residual
        return out



class dilated_residual_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3), stride=1, dilation_rate = [1,2,3]):
        super(dilated_residual_block, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mish = nn.Mish()
        self.conv1 = nn.Conv2d(input_channels, output_channels//4 , 1)
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, padding='same', dilation=dilation_rate[0])
        self.conv3 = nn.Conv2d(output_channels//4, output_channels//4, 3, padding='same', dilation=dilation_rate[1])
        self.conv4 = nn.Conv2d(output_channels//4, output_channels//4, 3, padding='same', dilation=dilation_rate[2])
        
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = True)
        self.conv6 = nn.Conv2d(input_channels, output_channels , 1, 1, padding='same', bias = True)
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.mish(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.mish(out)
        out1 = self.conv2(out)
        out1 = self.dropout1(out1)
        out2 = self.conv3(out)
        out2 = self.dropout2(out2)
        out3 = self.conv4(out)
        out3 = self.dropout3(out3)
        out = torch.add(torch.add(out1,out2),out3)
        out = self.bn3(out)
        out = self.mish(out)
        out = self.conv5(out)
        residual = self.conv6(x)
        out += residual
        return out
    


#---------------------------------ATTENTION MODULE 1----------------------------------#
class am1(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate = [4,8,12]):
        super(am1, self).__init__()

        # FIRST RESIDUAL BLOCKS
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)


        # TRUNK 
        self.trunk_branches = nn.Sequential(dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate), dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate))


        # ATTENTION MASK
        # Encoder
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.softmax1_blocks = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)
        self.skip1_connection_residual_block = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.softmax2_blocks = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)
        self.skip2_connection_residual_block = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate), dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate))
        self.skip3_connection_residual_block = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)  

        self.op1 = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate) 
        self.op2 = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate) 

        # Decoder
        self.interpolation3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax4_blocks = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.interpolation2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax5_blocks = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.interpolation1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax6_blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Sigmoid())


        # LAST RESIDUAL BLOCKS
        self.last_blocks = ResidualBlock(in_channels, out_channels)


    def forward(self, x):

        # FIRST RESIDUAL BLOCKS
        x = self.first_residual_blocks(x)


        # TRUNK BRANCH
        out_trunk = self.trunk_branches(x)


        # ATTENTION MASK
        # Encoder
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)

        out_op1 = self.op1(out_softmax3) 
        out_bottleneck = torch.add(out_op1, out_skip3_connection) 
        out_op2 = self.op2(out_bottleneck)

        # Decoder
        out_interp3 = self.interpolation3(out_op2) 
        out = torch.add(out_interp3, out_skip2_connection)
        out_softmax4 = self.softmax4_blocks(out)

        out_interp2 = self.interpolation2(out_softmax4)
        out = torch.add(out_interp2, out_skip1_connection)
        out_softmax5 = self.softmax5_blocks(out)

        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = torch.multiply((1 + out_softmax6), out_trunk)


        # LAST RESIDUAL BLOCKS
        out_last = self.last_blocks(out)


        return out_last
#---------------------------------------------------------------------------------------------------------------------#



#---------------------------------ATTENTION MODULE 2----------------------------------#
class am2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate = [2,4,6]):
        super(am2, self).__init__()

        # FIRST RESIDUAL BLOCKS
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)


        # TRUNK
        self.trunk_branches = nn.Sequential(dilated_residual_block(in_channels, out_channels, dilation_rate), dilated_residual_block(out_channels, out_channels, dilation_rate))


        # ATTENTION MASK
        # Encoder
        self.mpool1 = nn.MaxPool2d(kernel_size = 2, stride=2, padding=0)
        self.softmax1_blocks = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)
        self.skip1_connection_residual_block = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.softmax2_blocks = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate) 
        self.skip2_connection_residual_block = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate) 
        
        self.op0 = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate) 
        self.op1 = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate) 
        self.op2 = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate) 

        # Decoder
        self.interpolation2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax3_blocks = dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate)

        self.interpolation1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax4_blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Sigmoid())


        # LAST RESIDUAL BLOCKS
        self.last_blocks = ResidualBlock(in_channels, out_channels)


    def forward(self, x):

        # FIRST RESIDUAL BLOCKS
        x = self.first_residual_blocks(x)


        # TRUNK BRANCH
        out_trunk = self.trunk_branches(x)


        # ATTENTION MASK
        # Encoder
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2_1 = self.softmax2_blocks(out_mpool2)
        out_softmax2 = self.op0(out_softmax2_1)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2_1)

        out_op1 = self.op1(out_softmax2)
        out_bottleneck = torch.add(out_op1, out_skip2_connection)
        out_op2 = self.op2(out_bottleneck)
        

        # Decoder
        out_interp2 = self.interpolation2(out_op2)
        out = torch.add(out_interp2, out_skip1_connection)
        out_softmax3 = self.softmax3_blocks(out)

        out_interp1 = self.interpolation1(out_softmax3)
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = torch.multiply((1 + out_softmax4), out_trunk)


        # LAST RESIDUAL BLOCKS:
        out_last = self.last_blocks(out)

        return out_last
#---------------------------------------------------------------------------------------------------------------------#



#---------------------------------ATTENTION MODULE 3----------------------------------#
class am3(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate = [2,4,6]):
        super(am3, self).__init__()

        # FIRST RESIDUAL BLOCKS
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        
        # TRUNK BRANCH
        self.trunk_branches = nn.Sequential(dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate),dilated_residual_block(in_channels, out_channels, dilation_rate=dilation_rate))


        # ATTENTION MASK
        # Encoder
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)  
        self.skip1_connection_residual_block = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)

        self.op0 = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)
        self.op1 = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)
        self.op2 = dilated_residual_block(in_channels, out_channels,dilation_rate=dilation_rate)

        # Decoder
        self.interpolation1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax4_blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), nn.Sigmoid())


        # FINAL RESIDUAL BLOCKS
        self.last_blocks = ResidualBlock(in_channels, out_channels)


    def forward(self, x):

        # FIRST RESIDUAL BLOCKS
        x = self.first_residual_blocks(x)


        # TRUNK
        out_trunk = self.trunk_branches(x)


        # ATTENTION MASK
        #Encoder
        out_mpool1 = self.mpool1(x)
        out_softmax1_1 = self.softmax1_blocks(out_mpool1)
        out_softmax1 = self.op0(out_softmax1_1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1_1)
        
        out_op1 = self.op1(out_softmax1)
        out_bottleneck = torch.add(out_op1, out_skip1_connection)
        out_op2 = self.op2(out_bottleneck)


        # Decoder
        out_interp1 = self.interpolation1(out_op2)
        out_softmax2 = self.softmax4_blocks(out_interp1)
        out = torch.multiply((1 + out_softmax2), out_trunk)


        # LAST RESIDUAL BLOCKS
        out_last = self.last_blocks(out)

        return out_last
#---------------------------------------------------------------------------------------------------------------------#


#--------------------------------- MEDIANET-69 ------------------------------#
class Medianet69(nn.Module):
    # for input size 224
    def __init__(self,  numclasses = 13, n_channels=32):
        super(Medianet69, self).__init__()
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=7,  stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.mish1 = nn.Mish()
        self.mpool1 = nn.MaxPool2d(kernel_size=2)


        self.residual_block1 = ResidualBlock(n_channels, n_channels,stride=1)
        self.attention_module2_1 = am1(n_channels, n_channels, dilation_rate = [4,8,12])


        self.residual_block2 = ResidualBlock(n_channels, 2*n_channels, stride=2)
        self.attention_module2_2 = am2(2*n_channels, 2*n_channels,  dilation_rate = [2,4,6])


        self.residual_block3 = ResidualBlock(2*n_channels, 4*n_channels, stride=2)
        self.attention_module2_3 = am3(4*n_channels, 4*n_channels,  dilation_rate = [1,2,3])


        self.residual_block4 = ResidualBlock(4*n_channels, 8*n_channels, stride=2)    
        self.residual_block5 = ResidualBlock(8*n_channels, 256)  
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)  
        self.flatten = nn.Flatten() 
        self.dropout = nn.Dropout(0.25)
        self.dense1 = nn.Linear(256,2*n_channels) 
        self.mish2 = nn.Mish()
        self.fc2 = nn.Linear(2*n_channels, numclasses)


    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.mpool1(out)


        out = self.residual_block1(out)
        out = self.attention_module2_1(out)


        out = self.residual_block2(out)
        out = self.attention_module2_2(out)


        out = self.residual_block3(out)
        out = self.attention_module2_3(out)


        out = self.residual_block4(out)
        out = self.residual_block5(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.dropout(out)

        out = self.dense1(out)
        out = self.mish2(out)

        out = self.fc2(out)

        return out
#---------------------------------------------------------------------------------------------------------------------#



#--------------------------------- MEDIANET-117 ------------------------------#
class Medianet117(nn.Module):
    def __init__(self,  numclasses = 13, n_channels=32):
        super(Medianet117, self).__init__()
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=7,  stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.mish1 = nn.Mish()
        self.mpool1 = nn.MaxPool2d(kernel_size=2)


        self.residual_block1 = ResidualBlock(n_channels, n_channels,stride=1)
        self.attention_module2_1 = am1(n_channels, n_channels, dilation_rate = [4,8,12])


        self.residual_block2 = ResidualBlock(n_channels, 2*n_channels, stride=2)
        self.attention_module2_2 = am2(2*n_channels, 2*n_channels,  dilation_rate = [2,4,6])
        self.attention_module2_2_2 = am2(2*n_channels, 2*n_channels,  dilation_rate = [2,4,6])


        self.residual_block3 = ResidualBlock(2*n_channels, 4*n_channels, stride=2)
        self.attention_module2_3 = am3(4*n_channels, 4*n_channels,  dilation_rate = [1,2,3])
        self.attention_module2_3_3 = am3(4*n_channels, 4*n_channels,  dilation_rate = [1,2,3])
        self.attention_module2_3_3_3 = am3(4*n_channels, 4*n_channels,  dilation_rate = [1,2,3])


        self.residual_block4 = ResidualBlock(4*n_channels, 8*n_channels, stride=2)    
        self.residual_block5 = ResidualBlock(8*n_channels, 8*n_channels)    
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)  
        self.flatten = nn.Flatten() 
        self.dense1 = nn.Linear(256,2*n_channels) 
        self.dropout = nn.Dropout(0.25)
        self.mish2 = nn.Mish()
        self.fc2 = nn.Linear(2*n_channels, numclasses)

        self.residual_block6 = ResidualBlock(8*n_channels,256) 


    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.mpool1(out)


        out = self.residual_block1(out)
        out = self.attention_module2_1(out)


        out = self.residual_block2(out)
        out = self.attention_module2_2(out)
        out = self.attention_module2_2_2(out)


        out = self.residual_block3(out)
        out = self.attention_module2_3(out)
        out = self.attention_module2_3_3(out)
        out = self.attention_module2_3_3_3(out)


        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)  

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.dropout(out)

        out = self.dense1(out)
        out = self.mish2(out)

        out = self.fc2(out)

        return out
#---------------------------------------------------------------------------------------------------------------------#