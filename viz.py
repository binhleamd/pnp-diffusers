import sys
import torch
import torchvision.models as models
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter

def print_network_structure(model, indent=''):
    """
    Print a text representation of the network structure.
    
    Args:
        model (nn.Module): The PyTorch model to visualize
        indent (str): Current indentation level
    """
    # Get the model's children and count parameters
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{indent}├─ {name}: {child.__class__.__name__} ({params:,} parameters)")
        
        # Recursively print children with increased indentation
        print_network_structure(child, indent + '│  ')

def display_model_summary(model, args=None, kwargs=None):
    """
    Display a text-based summary of the model including number of parameters and layer details.
    
    Args:
        model (nn.Module): The PyTorch model
        input_size (tuple): Input size for parameter calculation
    """
    print(f"Model: {model.__class__.__name__}")
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
    # Try to compute output shape if possible
    try:
        model.eval()
        with torch.no_grad():
            out=model(*args, **kwargs)
        
        if isinstance(out, torch.Tensor):
            #print(f"Input shape: {tuple(out.shape)}")
            print(f"Output shape: {tuple(out.shape)}")
        else:
            print("Model has multiple or non-tensor outputs")
    except Exception as e:
        print(f"Couldn't compute output shape: {e}")

def visualize_network_using_torchviz(model, args=None, kwargs=None):
    """
    Visualize a PyTorch network using torchviz.
    
    Args:
        model (nn.Module): The PyTorch model to visualize
        input_size (tuple): The input size to use for the forward pass
    
    Returns:
        GraphViz dot graph
    """
    y=model(*args, **kwargs)
    
    # If model outputs a tuple/list, use the first element
    if isinstance(y, tuple) or isinstance(y, list):
        y = y[0]
    
    # Handle multiple outputs by summing them
    if isinstance(y, dict):
        y = sum(y.values())
    elif y.shape[1] > 1:
        y = y.sum()
    
    # Create the dot graph
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    return dot

# Example usage
if __name__ == "__main__":
    model = models.resnet18(weights=None)

    input_size=(1, 3, 224, 224)
    input = torch.randn(input_size)
    
    writer=SummaryWriter('viz/resnet18')
    writer.add_graph(model, input)
    writer.close()




    # with open('model_architecture.txt', 'w') as f:
    #     # Store the original stdout
    #     original_stdout = sys.stdout
    #     # Redirect stdout to the file
    #     sys.stdout = f
        
    #     # All print statements will now write to output.txt
    #     display_model_summary(model)
    #     print_network_structure(model)


    #     # Restore stdout to its original value
    #     sys.stdout = original_stdout

    #     print("Model architect saved as 'model_architecture.txt'")

    # visualize_network_using_torchviz(model).render('model_architecture', format='png', cleanup=True)
    # print("Torchviz visualization saved as 'model_architecture.png'")

