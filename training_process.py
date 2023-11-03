import torch
import torch.optim as optim
import data_loader
import model

def train_model(model, train_loader, epochs=5, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Mini-batch {i + 1}, Loss: {running_loss / 100}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), 'cifar10_model.pth')

if __name__ == "__main__":
    train_loader = data_loader.get_cifar10_train_loader()
    cnn_model = model.SimpleCNN()
    train_model(cnn_model, train_loader)
