import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np

# 1. Preparación de Datos (Usando el archivo Gold Standard)
print("Cargando el dataset Gold Standard...")
df = pd.read_csv('colorimetry_master_index.csv')

# Seleccionamos las 5 features biométricas
X = df[['Skin_L', 'Skin_b', 'Chroma', 'Iris_L', 'Iris_b']].values
le = LabelEncoder()
y = le.fit_transform(df['Season_12'])
NUM_CLASSES = len(le.classes_)
print(f"Clases detectadas: {NUM_CLASSES} ({le.classes_})")

# Escalado (Fundamental para Redes Neuronales)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Conversión a Tensores
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 2. Arquitectura de la Red (ColorNet)
class ColorNet(nn.Module):
    def __init__(self, num_classes):
        super(ColorNet, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Configuración de Hardware (CORREGIDO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorNet(num_classes=NUM_CLASSES).to(device)
print(f"Entrenando en: {device}")

# 3. Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Iniciando entrenamiento por lotes (Mini-batch GD)...")
epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            _, predicted = torch.max(test_outputs, 1)
            acc = (predicted == y_test.to(device)).sum().item() / y_test.size(0)
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Val Accuracy: {acc:.2%}")

# 4. Guardar resultados
torch.save(model.state_dict(), 'biometric_color_model.pth')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ ¡Modelo profesional entrenado y guardado!")
