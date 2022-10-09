from network import *
optimiser = torch.optim.Adam(EdgeDetector.parameters(), lr=0.001)
loss_function = nn.MSELoss()
loss_list = []

def train(epochs,train_dataloader,train_label_dataloader,EdgeDetector,loss_function,loss_list,optimiser):
    for epoch in range(epochs):
        print("Epoch:",epoch)
        cost = 0
        for i, (imgs, Y) in enumerate(zip(train_dataloader, train_label_dataloader)):
            a,b,c,d = imgs.size()
            imgs = imgs.reshape(a,b*c*d)
            output = EdgeDetector(imgs.to(device))
            optimiser.zero_grad()
            output = output.reshape(a,b,c,d)
            loss = loss_function(output,Y.to(device))
            cost = cost + loss.item()
            loss.backward()
            optimiser.step()
        avg_loss = cost/9999
        loss_list.append(avg_loss)
        print("For epoch: ", epoch, " the loss is :", avg_loss)
    return loss_list

cost_list = train(epochs,train_dataloader,train_label_dataloader,EdgeDetector,loss_function,loss_list,optimiser)
torch.save(EdgeDetector.state_dict(),"D:\\ML Mini project\\weights1.pth")


# for params in EdgeDetector.parameters():
#     params.requires_grad = False

# plt.plot(cost_list)
# plt.show()