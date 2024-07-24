from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import base64
import struct
import torch.nn.functional as F
from flask_cors import CORS
from torchvision.io import image, ImageReadMode



app = Flask(__name__)
CORS(app)


torch.set_printoptions(threshold=5000)
torch.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=50)

class Simple_Model:
    def __init__(self, input_dim, output_dim, num_layers, layer_widths, activation_function, norm_type, steps, learning_rate, batch_size, device="cpu") -> None:
        if(len(layer_widths) != num_layers):
            print(len(layer_widths))
            raise Exception("Your number of layer widths must be the same as the number of layers passed in") 
        layer_widths.insert(0, input_dim)
        layer_widths.append(output_dim)
        layers = []
        for i in range(num_layers + 1):
            linear_layer = Linear(layer_widths[i], layer_widths[i + 1], device=device)
            if(norm_type == "Batch"):
                norm_layer = BatchNorm1d(layer_widths[i + 1], device=device)
            elif(norm_type == "Layer"):
                norm_layer = LayerNorm(layer_widths[i + 1], device=device)
            if(activation_function == "Tanh"):
                activation = Tanh()
            elif(activation_function == "Relu"):
                activation = Relu()
            if(i == num_layers):
                layers.append(linear_layer)
            else:
                layers = layers + [linear_layer, norm_layer, activation]
        self.layers = layers
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.model = Sequential(layers)
        self.lossi = []
        self.input_dim = input_dim
        for p in self.model.parameters():
            p.requires_grad = True

    def train(self, x, y, train_print_step=100):
        for i in range(self.steps):
            xb, yb = self.get_batch(x, y)
            xb = xb.view(self.batch_size, -1)
            for layer in self.layers:
                if(type(layer) is BatchNorm1d):
                    xb = layer(xb, training=True)
                else:
                    xb = layer(xb)
            loss = F.cross_entropy(xb, yb)
            for p in self.model.parameters():
                p.grad = None
            loss.backward()

            for p in self.model.parameters():
                p.data += -self.learning_rate * p.grad

            if i % train_print_step == 0:
                print(f'{i:7d}/{self.steps:7d}: {loss.item():.4f}')
        
            self.lossi.append(loss.log10().item())

    def get_batch(self, x, y):
        indexes = torch.randint(x.shape[0], size=(self.batch_size,))
        x_batch = x[indexes]
        y_batch = y[indexes]
        return x_batch, y_batch
    
    def plot_lossi(self):
        plt.plot(self.lossi)

    def sample_from_model(self, x, debug=False):
        x = x.view(-1, self.input_dim).to(self.device)

        xcopy = x.clone()
        if(debug):
            print("copy:")
            np.set_printoptions(threshold=np.inf)
            print(xcopy.detach().cpu().numpy())

            print(self.layers)

        for layer in self.layers:
            x = layer(x)
            if(debug):
                print(x)


        x = torch.softmax(x, dim=1)

        y_pred_index = torch.argmax(x, dim=1).item()

        print(f"Greatest Probability: {y_pred_index}")
        return y_pred_index

    def calculate_loss(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x = x.view(len(x), -1)
        for layer in self.layers:
            x = layer(x)
        loss = F.cross_entropy(x, y)
        print(loss.item())
        print(x.shape)

        y_pred_index = torch.argmax(x, dim=1)
        print(y_pred_index)

        same = torch.sum(y_pred_index == y).item()
        diff = len(y) - same

        print(f"Correct: {same}")
        print(f"Wrong: {diff}")
        print(f"Proportion: {same / len(y)}")


    def visualize_histograms(self):
        pass

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, device='cpu'):
        # initialize weights and biases
        self.weights = torch.randn((fan_in, fan_out), device=device)
        self.biases = torch.randn((1, fan_out), device=device) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weights
        if self.biases is not None:
            self.out = self.out + self.biases
        return self.out
        
    def parameters(self):
        return [self.weights] + ([] if self.biases is None else [self.biases])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, device='cpu'):
        self.eps = eps
        self.momentum = momentum
        self.training = True
          
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
          
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)
        
    def __call__(self, x, training=False):
        if training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        with torch.no_grad():
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar

        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm:
    def __init__(self, dim, eps=1e-5, device='cpu'):
        self.eps = eps
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)

    def __call__(self, x):
        layer_mean = x.mean(1, keepdim=True)
        layer_var = x.var(1, keepdim=True)
        xhat = (x - layer_mean) / torch.sqrt(layer_var + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Relu:
    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return torch.from_numpy(data.reshape((size, nrows, ncols))).float()

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return torch.from_numpy(data.reshape((size,)))

# Device configuration
device = torch.device("mps")

# Load training and test data
x = load_mnist_images('train-images.idx3-ubyte').to(device)
y = load_mnist_labels('train-labels.idx1-ubyte').to(device)
xtest = load_mnist_images('t10k-images.idx3-ubyte').to(device)
ytest = load_mnist_labels('t10k-labels.idx1-ubyte').to(device)

# Initialize and train model
model = Simple_Model(input_dim=784, output_dim=10, num_layers=4, layer_widths=[768, 768, 512, 512], activation_function="Tanh", norm_type="Batch", steps=100, learning_rate=0.01, batch_size=256, device=device)
model.train(x, y, train_print_step=250)

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    
    # Open the image and convert it to grayscale
    image = Image.open(io.BytesIO(image_data)).convert('L')

    image.save("received_image.png", "PNG")


    # Save the image to a file
    # with open("received_image.png", "wb") as f:
    #     f.write(image)

    
    # Resize the image to 28x28
    image = image.resize((28, 28))
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    print("Unique values in image array:", np.unique(image_array))
    print("Min value:", np.min(image_array))
    print("Max value:", np.max(image_array))
    
    # Normalize the image (values between 0 and 1)
    image_array = image_array / 255.0
    
    # Invert the colors (1 - x)
    image_array = 1 - image_array
    
    # Convert to tensor and reshape
    image_tensor = torch.tensor(image_array, dtype=torch.float32).reshape(1, 784)
    
    # Move tensor to the correct device
    image_tensor = image_tensor.to(device)

    print("Tensor shape:", image_tensor.shape)
    print("Tensor min:", torch.min(image_tensor))
    print("Tensor max:", torch.max(image_tensor))

    # Get prediction from the model
    prediction = model.sample_from_model(image_tensor)
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('index.html')