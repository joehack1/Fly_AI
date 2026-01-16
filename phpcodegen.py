# php_ai.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

print("🚀 Building PHP Code Assistant AI...")

# Sample PHP code patterns for training
php_examples = [
    # [Input: description, Output: PHP code]
    ["connect to mysql database", 
     "<?php\n$conn = new mysqli('localhost', 'username', 'password', 'database');\nif ($conn->connect_error) {\n    die('Connection failed: ' . $conn->connect_error);\n}\necho 'Connected successfully';?>"],
    
    ["validate email address",
     "<?php\nfunction validateEmail($email) {\n    return filter_var($email, FILTER_VALIDATE_EMAIL);\n}\n?>"],
    
    ["read file content",
     "<?php\n$content = file_get_contents('file.txt');\necho $content;\n?>"],
    
    ["write to file",
     "<?php\n$file = fopen('file.txt', 'w');\nfwrite($file, 'Hello World');\nfclose($file);\n?>"],
    
    ["create login form",
     "<?php\nsession_start();\nif ($_SERVER['REQUEST_METHOD'] == 'POST') {\n    $username = $_POST['username'];\n    $password = $_POST['password'];\n    // Add authentication logic here\n}\n?>\n<form method='POST'>\n    <input type='text' name='username'>\n    <input type='password' name='password'>\n    <button type='submit'>Login</button>\n</form>"],
    
    ["array to json",
     "<?php\n$array = ['name' => 'John', 'age' => 30];\necho json_encode($array);\n?>"],
    
    ["fetch api data",
     "<?php\n$url = 'https://api.example.com/data';\n$data = file_get_contents($url);\n$result = json_decode($data, true);\nprint_r($result);\n?>"],
    
    ["database select query",
     "<?php\n$sql = 'SELECT * FROM users WHERE active = 1';\n$result = $conn->query($sql);\nwhile($row = $result->fetch_assoc()) {\n    echo $row['username'];\n}\n?>"],
    
    ["upload file",
     "<?php\nif(isset($_FILES['file'])) {\n    $target_dir = 'uploads/';\n    $target_file = $target_dir . basename($_FILES['file']['name']);\n    move_uploaded_file($_FILES['file']['tmp_name'], $target_file);\n}\n?>"],
    
    ["send email",
     "<?php\n$to = 'user@example.com';\n$subject = 'Subject';\n$message = 'Hello!';\n$headers = 'From: webmaster@example.com';\nmail($to, $subject, $message, $headers);\n?>"]
]

# Build global vocabulary
all_texts = [desc + ' ' + code for desc, code in php_examples]
all_tokens = set()
for text in all_texts:
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    all_tokens.update(tokens)
vocab = {token: idx + 1 for idx, token in enumerate(sorted(all_tokens))}  # 0 for padding/unknown
reverse_vocab = {idx: token for token, idx in vocab.items()}

# Simple tokenizer
def tokenize(text):
    # Convert text to lowercase and split
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    # Convert to IDs using global vocab
    token_ids = [vocab.get(token, 0) for token in tokens]
    
    # Pad/truncate to fixed length
    max_len = 50
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    
    return token_ids

class PHPAIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        desc, code = self.examples[idx]
        
        # Tokenize input (description) and output (code)
        input_ids = tokenize(desc)
        output_ids = tokenize(code)
        
        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'output': torch.tensor(output_ids, dtype=torch.long)
        }

# Simple Sequence-to-Sequence model
class PHPGenerator(nn.Module):
    def __init__(self, vocab_size=len(vocab) + 1, embed_size=64, hidden_size=128):
        super(PHPGenerator, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_seq, output_seq=None, max_len=50):
        # Encode input
        embedded = self.embedding(input_seq)
        encoder_output, (hidden, cell) = self.encoder_lstm(embedded)
        
        # If training, use teacher forcing
        if output_seq is not None:
            decoder_input = self.embedding(output_seq)
            decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            return output
        
        # If inference, generate token by token
        else:
            generated = []
            # Start with SOS token (0)
            current_input = torch.tensor([[0]], dtype=torch.long).to(input_seq.device)
            
            for _ in range(max_len):
                embedded = self.embedding(current_input)
                decoder_output, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
                output = self.fc(decoder_output)
                _, next_token = torch.max(output, dim=-1)
                generated.append(next_token.item())
                current_input = next_token
            
            return generated

# Train the model
def train_php_ai():
    dataset = PHPAIDataset(php_examples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = PHPGenerator()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("📚 Training PHP Code Generator...")
    
    for epoch in range(100):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_seq = batch['input']
            target_seq = batch['output']
            
            # Forward pass
            output = model(input_seq, target_seq[:, :-1])
            
            # Calculate loss
            loss = criterion(output.transpose(1, 2), target_seq[:, 1:])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Test the model
    print("\n🧪 Testing PHP Generator:")
    test_inputs = [
        "connect to database",
        "validate email",
        "read file"
    ]
    
    for test_input in test_inputs:
        input_ids = tokenize(test_input)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            generated_ids = model(input_tensor, output_seq=None)
        
        # Decode generated tokens
        generated_tokens = [reverse_vocab.get(id, '') for id in generated_ids if id != 0]
        generated_code = ' '.join(generated_tokens)
        # Simple formatting
        generated_code = generated_code.replace(' <?php', '<?php').replace(' \n ', '\n').replace(' \n', '\n').replace(' ?>', '?>\n')
        
        print(f"\n💭 Input: {test_input}")
        print(f"📝 Generated:\n{generated_code}")
    
    return model

# Save the model
model = train_php_ai()
torch.save(model.state_dict(), 'php_generator.pth')
print("\n💾 Model saved as 'php_generator.pth'")