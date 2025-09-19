# Adding New Encoder Guide

This guide explains how to add a new encoder to the PK/PD modeling system.

## 📋 **Overview**

To add a new encoder, you need to modify **4 files**:
1. `models/encoders.py` - Implement the encoder class
2. `utils/helpers.py` - Add to factory function
3. `config.py` - Add configuration parameters
4. `README.md` - Update documentation

## 🚀 **Step-by-Step Guide**

### **Step 1: Implement Encoder Class**

Add your encoder class to `models/encoders.py`:

```python
class YourNewEncoder(BaseEncoder):
    """Your new encoder description"""
    
    def __init__(
        self, 
        in_dim: int, 
        hidden: int = 64, 
        depth: int = 3, 
        dropout: float = 0.1,
        # Add your custom parameters here
        your_param1: int = 10,
        your_param2: float = 0.5
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden
        self.depth = depth
        self.your_param1 = your_param1
        self.your_param2 = your_param2
        
        # Your encoder architecture
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Final projection
        self.final_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor [batch_size, in_dim] or [batch_size, seq_len, in_dim]
        Returns:
            Output tensor [batch_size, hidden]
        """
        # Handle sequence input if needed
        if x.dim() == 3:
            x = x.mean(dim=1)  # Global average pooling
        
        # Your forward logic
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_proj(x)
        x = self.dropout(x)
        
        return x
```

**Important Notes:**
- Inherit from `BaseEncoder`
- Set `self.in_dim` and `self.out_dim` attributes
- Handle both 2D `[batch_size, in_dim]` and 3D `[batch_size, seq_len, in_dim]` inputs
- Return tensor with shape `[batch_size, hidden]`

### **Step 2: Add to Factory Function**

Add your encoder to `utils/helpers.py` in the `build_encoder` function:

```python
def build_encoder(encoder_type, input_dim, config):
    """Build encoder"""
    if encoder_type == "mlp":
        # ... existing code ...
    
    elif encoder_type == "your_encoder_name":  # Add this block
        from models.encoders import YourNewEncoder
        return YourNewEncoder(
            in_dim=input_dim,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            your_param1=getattr(config, 'your_param1', 10),
            your_param2=getattr(config, 'your_param2', 0.5)
        )
    
    else:
        # Fallback to MLP for unknown encoders
        from models.encoders import MLPEncoder
        return MLPEncoder(input_dim, config.hidden, config.depth, config.dropout)
```

**Important Notes:**
- Use `getattr(config, 'param_name', default_value)` for optional parameters
- Import your encoder class inside the function to avoid circular imports

### **Step 3: Add Configuration Parameters**

Add your encoder parameters to `config.py`:

#### **3.1. Add to Config dataclass:**

```python
@dataclass
class Config:
    # ... existing parameters ...
    
    # === Your Encoder settings ===
    your_param1: int = 10
    your_param2: float = 0.5
```

#### **3.2. Add to argument parser:**

```python
def create_argument_parser():
    parser = argparse.ArgumentParser(description="PK/PD Modeling")
    # ... existing arguments ...
    
    # === Your Encoder settings ===
    parser.add_argument("--your_param1", type=int, default=10, help="Your parameter 1")
    parser.add_argument("--your_param2", type=float, default=0.5, help="Your parameter 2")
```

#### **3.3. Add to parse_args function:**

```python
def parse_args():
    args = create_argument_parser().parse_args()
    return Config(
        # ... existing parameters ...
        
        # Your encoder settings
        your_param1=args.your_param1,
        your_param2=args.your_param2,
    )
```

#### **3.4. Add to encoder choices:**

```python
parser.add_argument("--encoder", 
                   choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe", "cnn", "your_encoder_name"], 
                   default="mlp", help="Default encoder type")
parser.add_argument("--encoder_pk", 
                   choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe", "cnn", "your_encoder_name"], 
                   default=None, help="PK-specific encoder type")
parser.add_argument("--encoder_pd", 
                   choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe", "cnn", "your_encoder_name"], 
                   default=None, help="PD-specific encoder type")
```

### **Step 4: Update Documentation**

Update `README.md` to include your new encoder:

#### **4.1. Add to Advanced Encoders section:**

```markdown
### **Advanced Encoders**
- **MLP**: Multi-layer perceptron
- **ResMLP**: Residual MLP with skip connections
- **MoE**: Mixture of Experts
- **ResMLP+MoE**: Combined residual and mixture of experts
- **Adaptive ResMLP+MoE**: Adaptive mixture of experts
- **CNN**: Convolutional neural network for sequence data
- **Your Encoder**: Description of your encoder
```

#### **4.2. Add to Encoders section:**

```markdown
### **Encoders**
- `mlp`: Standard multi-layer perceptron
- `resmlp`: Residual MLP with skip connections
- `moe`: Mixture of Experts
- `resmlp_moe`: Combined ResMLP and MoE
- `adaptive_resmlp_moe`: Adaptive mixture of experts
- `cnn`: Convolutional neural network for sequence data
- `your_encoder_name`: Description of your encoder
```

#### **4.3. Add to Key Parameters section:**

```markdown
--encoder {mlp,resmlp,moe,resmlp_moe,adaptive_resmlp_moe,cnn,your_encoder_name}
```

#### **4.4. Add to Advanced Features section:**

```markdown
### **Advanced Features**
```bash
--use_fe                   # Enable feature engineering
--use_mixup                # Enable mixup augmentation
--lambda_contrast LAMBDA   # Contrastive learning weight
--temperature TEMP         # Temperature for contrastive learning
--kernel_size SIZE         # CNN kernel size
--num_filters FILTERS      # Number of CNN filters
--your_param1 VALUE        # Your parameter 1
--your_param2 VALUE        # Your parameter 2
```
```

## 📝 **Complete Example: TransformerEncoder**

Here's a complete example of adding a TransformerEncoder:

### **1. Encoder Implementation**

```python
class TransformerEncoder(BaseEncoder):
    """Transformer-based encoder for PK/PD modeling"""
    
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, hidden))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection
        self.output_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Handle 2D input (no sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, in_dim]
        
        seq_len = x.size(1)
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, hidden]
        
        # Add positional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        else:
            x = x + self.pos_encoding[:self.max_seq_len].unsqueeze(0)
            x = x[:, :self.max_seq_len]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, seq_len, hidden]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, hidden]
        
        # Output projection
        x = self.output_proj(x)
        x = self.dropout(x)
        
        return x
```

### **2. Factory Function**

```python
elif encoder_type == "transformer":
    from models.encoders import TransformerEncoder
    return TransformerEncoder(
        in_dim=input_dim,
        hidden=config.hidden,
        depth=config.depth,
        num_heads=getattr(config, 'num_heads', 8),
        dropout=config.dropout,
        max_seq_len=getattr(config, 'max_seq_len', 100)
    )
```

### **3. Configuration**

```python
# Config dataclass
num_heads: int = 8
max_seq_len: int = 100

# Argument parser
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--max_seq_len", type=int, default=100, help="Maximum sequence length")

# Encoder choices
choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe", "cnn", "transformer"]

# Parse args
num_heads=args.num_heads,
max_seq_len=args.max_seq_len,
```

## 🧪 **Testing Your New Encoder**

After implementing your encoder, test it:

```bash
# Test with default parameters
python main.py --run_name test_new_encoder --mode separate --encoder your_encoder_name --epochs 5

# Test with custom parameters
python main.py --run_name test_new_encoder --mode separate --encoder your_encoder_name --your_param1 20 --your_param2 0.8 --epochs 5

# Test with different PK/PD encoders
python main.py --run_name test_mixed --mode joint --encoder_pk mlp --encoder_pd your_encoder_name --epochs 5
```

## ✅ **Verification Checklist**

- [ ] Encoder class inherits from `BaseEncoder`
- [ ] `self.in_dim` and `self.out_dim` are set
- [ ] Forward method handles both 2D and 3D inputs
- [ ] Factory function added to `build_encoder`
- [ ] Configuration parameters added to `Config` dataclass
- [ ] Argument parser updated with new parameters
- [ ] Encoder choices updated in all encoder arguments
- [ ] `parse_args` function includes new parameters
- [ ] README.md updated with new encoder information
- [ ] Test runs successfully with new encoder

## 🚨 **Common Issues**

### **Issue 1: ImportError**
```
ImportError: cannot import name 'YourEncoder' from 'models.encoders'
```
**Solution:** Make sure your encoder class is properly defined in `models/encoders.py`

### **Issue 2: TypeError in __init__**
```
TypeError: YourEncoder.__init__() takes 1 positional argument but 3 were given
```
**Solution:** Check BaseEncoder inheritance and make sure you call `super().__init__()` correctly

### **Issue 3: Unknown encoder type**
```
Unknown encoder type: your_encoder_name
```
**Solution:** Make sure you added your encoder to the factory function in `utils/helpers.py`

### **Issue 4: Configuration not found**
```
AttributeError: 'Config' object has no attribute 'your_param'
```
**Solution:** Make sure you added the parameter to the Config dataclass and parse_args function

## 🎯 **Best Practices**

1. **Follow naming conventions**: Use descriptive names like `TransformerEncoder`, `CNNEncoder`
2. **Handle input flexibility**: Support both 2D and 3D inputs
3. **Use getattr for optional parameters**: `getattr(config, 'param', default_value)`
4. **Add comprehensive docstrings**: Document your encoder's purpose and parameters
5. **Test thoroughly**: Test with different modes and parameter combinations
6. **Update documentation**: Keep README.md and this guide updated

## 📚 **Real Example: CNNEncoder Implementation**

Here's the actual CNNEncoder that was added to the system:

### **1. CNNEncoder Class (models/encoders.py)**

```python
class CNNEncoder(BaseEncoder):
    """CNN-based encoder for sequence data"""
    
    def __init__(
        self, 
        in_dim: int, 
        hidden: int = 64, 
        depth: int = 3, 
        dropout: float = 0.1,
        kernel_size: int = 3,
        num_filters: int = 64
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        # Input projection to create channels
        self.input_proj = nn.Linear(in_dim, num_filters)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            in_channels = num_filters if i == 0 else num_filters
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.final_proj = nn.Linear(num_filters, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Handle 2D input (add sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, in_dim]
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, num_filters]
        
        # Transpose for Conv1d: [batch_size, num_filters, seq_len]
        x = x.transpose(1, 2)
        
        # CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling: [batch_size, num_filters, 1]
        x = self.global_pool(x)
        
        # Flatten: [batch_size, num_filters]
        x = x.squeeze(-1)
        
        # Final projection: [batch_size, hidden]
        x = self.final_proj(x)
        x = self.dropout(x)
        
        return x
```

### **2. Factory Function (utils/helpers.py)**

```python
elif encoder_type == "cnn":
    from models.encoders import CNNEncoder
    return CNNEncoder(
        in_dim=input_dim,
        hidden=config.hidden,
        depth=config.depth,
        dropout=config.dropout,
        kernel_size=getattr(config, 'kernel_size', 3),
        num_filters=getattr(config, 'num_filters', 64)
    )
```

### **3. Configuration (config.py)**

```python
# Config dataclass
kernel_size: int = 3
num_filters: int = 64

# Argument parser
parser.add_argument("--kernel_size", type=int, default=3, help="CNN kernel size")
parser.add_argument("--num_filters", type=int, default=64, help="Number of CNN filters")

# Encoder choices
choices=["mlp", "resmlp", "moe", "resmlp_moe", "adaptive_resmlp_moe", "cnn"]

# Parse args
kernel_size=args.kernel_size,
num_filters=args.num_filters,
```

### **4. Usage Examples**

```bash
# Basic CNN encoder
python main.py --run_name cnn_test --mode separate --encoder cnn --epochs 10

# CNN with custom parameters
python main.py --run_name cnn_custom --mode separate --encoder cnn --kernel_size 5 --num_filters 128 --epochs 10

# Different PK/PD encoders
python main.py --run_name mixed_encoders --mode joint --encoder_pk mlp --encoder_pd cnn --epochs 10
```