from network import *
from model import *

test_dataloader = DataLoader(
    ImageDataset("D:\\FYP\\context_encoder\\img_align_celeba_10k", transforms_=transforms_, mode="val"),
    batch_size=5,
    shuffle=False,
) # TestSet DataLoader

for i,imgaa in enumerate(test_dataloader):
    k = imgaa
    
no_of_imgs = k.shape[0]
# #Before passing through AutoEncoder
print("before Passing through Network")
fig, axes = plt.subplots(1,no_of_imgs-1)
k_ = k
k_ = (k_.to(torch.device('cpu')).detach().numpy()).reshape(no_of_imgs,img_size,img_size)

axes[0].imshow(k_[0],cmap="gray")
axes[0].set_axis_off()
axes[1].imshow(k_[1],cmap="gray")
axes[1].set_axis_off()
axes[2].imshow(k_[2],cmap="gray")
axes[2].set_axis_off()

#After Passing through AutoEncoder
a,b,c,d = k.size()
k = k.reshape(a,b*c*d)
print("After Passing through Network")
outputk = EdgeDetector(k[0:no_of_imgs].to(device)).to(torch.device('cpu')).detach().numpy()
outputk = outputk.reshape(no_of_imgs,img_size,img_size)
fig, axes = plt.subplots(1,no_of_imgs-1)
axes[0].imshow(outputk[0],cmap="gray")
axes[0].set_axis_off()
axes[1].imshow(outputk[1],cmap="gray")
axes[1].set_axis_off()
axes[2].imshow(outputk[2],cmap="gray")
axes[2].set_axis_off()

plt.show()
