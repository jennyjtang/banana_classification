import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class BananaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)  
        self.fc2 = nn.Linear(128, 2)  # unripe vs ripe

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== Model 2: Average Pooling CNN ====================
class BananaCNN_AvgPool(nn.Module):
    """Uses Average Pooling instead of Max Pooling"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)  # Average pooling
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== Model 3: Mixed Pooling CNN ====================
class BananaCNN_MixedPool(nn.Module):
    """Uses both Max and Average pooling"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))  # Max for first layer
        x = self.avgpool(F.relu(self.conv2(x)))  # Avg for second
        x = self.maxpool(F.relu(self.conv3(x)))  # Max for third
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== Model 4: Deeper CNN with Batch Normalization ====================
class BananaCNN_Deep(nn.Module):
    """Deeper model with BatchNorm and Dropout"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ==================== Model 5: Strided Convolutions (No Pooling) ====================
class BananaCNN_Strided(nn.Module):
    """Uses strided convolutions instead of pooling layers"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  # stride=2 reduces size
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== Model 6: Global Average Pooling ====================
class BananaCNN_GlobalAvgPool(nn.Module):
    """Uses Global Average Pooling before classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global avg pooling
        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = self.fc(x)
        return x

# ==================== Model 7: ResNet-Style with Skip Connections ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BananaCNN_ResNet(nn.Module):
    """ResNet-style with skip connections"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==================== Model 8: VGG-Style (Deeper with small filters) ====================
class BananaCNN_VGG(nn.Module):
    """VGG-style with multiple 3x3 convolutions"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== Model 9: Transfer Learning - ResNet18 ====================
def get_resnet18_pretrained():
    """Pretrained ResNet18 from torchvision"""
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model

# ==================== Model 10: Transfer Learning - MobileNetV2 ====================
def get_mobilenet_pretrained():
    """Pretrained MobileNetV2 (lightweight for mobile deployment)"""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

# ==================== Comprehensive Evaluation Function ====================
def evaluate_model(model, test_loader, device='cuda', model_name='Model'):
    """
    Comprehensive evaluation with multiple metrics
    Returns: dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_unripe': precision[0],
        'recall_unripe': recall[0],
        'f1_unripe': f1[0],
        'precision_ripe': precision[1],
        'recall_ripe': recall[1],
        'f1_ripe': f1[1],
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results

# ==================== Print Results Function ====================
def print_results(results):
    """Pretty print evaluation results"""
    print(f"\n{'='*60}")
    print(f"Results for: {results['model_name']}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"\nPer-Class Metrics:")
    print(f"  Unripe - Precision: {results['precision_unripe']:.4f}, "
          f"Recall: {results['recall_unripe']:.4f}, F1: {results['f1_unripe']:.4f}")
    print(f"  Ripe   - Precision: {results['precision_ripe']:.4f}, "
          f"Recall: {results['recall_ripe']:.4f}, F1: {results['f1_ripe']:.4f}")
    print(f"\nMacro Averages:")
    print(f"  Precision: {results['precision_macro']:.4f}")
    print(f"  Recall: {results['recall_macro']:.4f}")
    print(f"  F1-Score: {results['f1_macro']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {results['confusion_matrix']}")
    print(f"{'='*60}\n")

# ==================== Plot Confusion Matrix ====================
def plot_confusion_matrix(results, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unripe', 'Ripe'],
                yticklabels=['Unripe', 'Ripe'])
    plt.title(f"Confusion Matrix - {results['model_name']}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ==================== Compare Multiple Models ====================
def compare_models(results_list):
    """
    Compare multiple models side by side
    results_list: list of result dictionaries from evaluate_model
    """
    print(f"\n{'='*100}")
    print(f"{'Model':<30} {'Accuracy':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12}")
    print(f"{'='*100}")
    
    for results in results_list:
        print(f"{results['model_name']:<30} "
              f"{results['accuracy']:<12.2f} "
              f"{results['f1_macro']:<12.4f} "
              f"{results['precision_macro']:<12.4f} "
              f"{results['recall_macro']:<12.4f}")
    
    print(f"{'='*100}\n")
    
    # Plot comparison
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    f1_scores = [r['f1_macro'] * 100 for r in results_list]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Percentage')
    ax.set_title('Model Comparison: Accuracy and F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== Example Usage ====================
def main():
    """Example of how to use all models and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assume you have a test_loader already created
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Dictionary of all models to test
    models_dict = {
        'Original MaxPool': BananaCNN(),
        'AvgPool': BananaCNN_AvgPool(),
        'Mixed Pool': BananaCNN_MixedPool(),
        'Deep BN+Dropout': BananaCNN_Deep(),
        'Strided Conv': BananaCNN_Strided(),
        'Global AvgPool': BananaCNN_GlobalAvgPool(),
        'ResNet-Style': BananaCNN_ResNet(),
        'VGG-Style': BananaCNN_VGG(),
        'ResNet18 Pretrained': get_resnet18_pretrained(),
        'MobileNetV2': get_mobilenet_pretrained()
    }
    
    results_list = []
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        # Load trained weights (you need to train these first)
        # model.load_state_dict(torch.load(f'{model_name}_weights.pth'))
        
        model = model.to(device)
        results = evaluate_model(model, test_loader, device, model_name)
        print_results(results)
        plot_confusion_matrix(results, save_path=f'{model_name}_cm.png')
        results_list.append(results)
    
    # Compare all models
    compare_models(results_list)

if __name__ == '__main__':
    main()