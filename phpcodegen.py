# php_ai.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import pickle

print("🚀 Building PHP Code Assistant AI...")

# Sample PHP code patterns for training
php_examples = [
    # [Input: description, Output: PHP code]
    ["connect to mysql database", 
     "<?php\n$conn = new mysqli('localhost', 'username', 'password', 'database');\nif ($conn->connect_error) {\n    die('Connection failed: ' . $conn->connect_error);\n}\necho 'Connected successfully';?>"],
    
    ["validate email address",
     "<?php\nfunction validateEmail($email) {\n    return filter_var($email, FILTER_VALIDATE_EMAIL);\n}\n?>"],

    ["check if email is valid",
     "<?php\nfunction isValidEmail($email) {\n    return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;\n}\n?>"],

    ["read file content",
     "<?php\n$content = file_get_contents('file.txt');\necho $content;\n?>"],

    ["read from file",
     "<?php\n$handle = fopen('file.txt', 'r');\n$content = fread($handle, filesize('file.txt'));\nfclose($handle);\necho $content;\n?>"],

    ["write to file",
     "<?php\n$file = fopen('file.txt', 'w');\nfwrite($file, 'Hello World');\nfclose($file);\n?>"],

    ["append to file",
     "<?php\n$file = fopen('file.txt', 'a');\nfwrite($file, 'New content');\nfclose($file);\n?>"],

    ["create login form",
     "<?php\nsession_start();\nif ($_SERVER['REQUEST_METHOD'] == 'POST') {\n    $username = $_POST['username'];\n    $password = $_POST['password'];\n    // Add authentication logic here\n}\n?>\n<form method='POST'>\n    <input type='text' name='username'>\n    <input type='password' name='password'>\n    <button type='submit'>Login</button>\n</form>"],

    ["create registration form",
     "<?php\nif ($_SERVER['REQUEST_METHOD'] == 'POST') {\n    $username = $_POST['username'];\n    $email = $_POST['email'];\n    $password = $_POST['password'];\n    // Process registration\n}\n?>\n<form method='POST'>\n    <input type='text' name='username'>\n    <input type='email' name='email'>\n    <input type='password' name='password'>\n    <button type='submit'>Register</button>\n</form>"],

    ["array to json",
     "<?php\n$array = ['name' => 'John', 'age' => 30];\necho json_encode($array);\n?>"],

    ["json to array",
     "<?php\n$json = '{\"name\":\"John\",\"age\":30}';\n$array = json_decode($json, true);\nprint_r($array);\n?>"],

    ["fetch api data",
     "<?php\n$url = 'https://api.example.com/data';\n$data = file_get_contents($url);\n$result = json_decode($data, true);\nprint_r($result);\n?>"],

    ["make http request",
     "<?php\n$url = 'https://api.example.com/data';\n$options = ['http' => ['method' => 'GET']];\n$context = stream_context_create($options);\n$result = file_get_contents($url, false, $context);\necho $result;\n?>"],

    ["database select query",
     "<?php\n$sql = 'SELECT * FROM users WHERE active = 1';\n$result = $conn->query($sql);\nwhile($row = $result->fetch_assoc()) {\n    echo $row['username'];\n}\n?>"],

    ["insert into database",
     "<?php\n$sql = 'INSERT INTO users (username, email) VALUES (?, ?)';\n$stmt = $conn->prepare($sql);\n$stmt->bind_param('ss', $username, $email);\n$stmt->execute();\n?>"],

    ["upload file",
     "<?php\nif(isset($_FILES['file'])) {\n    $target_dir = 'uploads/';\n    $target_file = $target_dir . basename($_FILES['file']['name']);\n    move_uploaded_file($_FILES['file']['tmp_name'], $target_file);\n}\n?>"],

    ["check file exists",
     "<?php\nif (file_exists('file.txt')) {\n    echo 'File exists';\n} else {\n    echo 'File does not exist';\n}\n?>"],

    ["send email",
     "<?php\n$to = 'user@example.com';\n$subject = 'Subject';\n$message = 'Hello!';\n$headers = 'From: webmaster@example.com';\nmail($to, $subject, $message, $headers);\n?>"],

    ["send html email",
     "<?php\n$to = 'user@example.com';\n$subject = 'HTML Email';\n$message = '<h1>Hello!</h1><p>This is HTML email.</p>';\n$headers = 'MIME-Version: 1.0' . \"\\r\\n\" . 'Content-type: text/html; charset=UTF-8' . \"\\r\\n\" . 'From: webmaster@example.com';\nmail($to, $subject, $message, $headers);\n?>"],

    ["get current date",
     "<?php\necho date('Y-m-d H:i:s');\n?>"],

    ["format date",
     "<?php\n$date = date('F j, Y');\necho $date;\n?>"],

    ["calculate age",
     "<?php\n$birthdate = '1990-01-01';\n$today = date('Y-m-d');\n$age = date_diff(date_create($birthdate), date_create($today));\necho $age->format('%y');\n?>"],

    ["generate random number",
     "<?php\necho rand(1, 100);\n?>"],

    ["hash password",
     "<?php\n$password = 'mypassword';\n$hashed = password_hash($password, PASSWORD_DEFAULT);\necho $hashed;\n?>"],

    ["verify password",
     "<?php\n$password = 'mypassword';\n$hash = '$2y$10$...';\nif (password_verify($password, $hash)) {\n    echo 'Valid password';\n}\n?>"]
]

# Build global vocabulary
all_texts = [desc + ' ' + code for desc, code in php_examples]
all_tokens = set()
for text in all_texts:
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    all_tokens.update(tokens)
vocab = {token: idx + 1 for idx, token in enumerate(sorted(all_tokens))}  # 0 for padding/unknown
reverse_vocab = {idx: token for token, idx in vocab.items()}

# Save vocab
with open('vocab.pkl', 'wb') as f:
    pickle.dump((vocab, reverse_vocab), f)

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

# Simple Sequence-to-Sequence model with Attention
class PHPGenerator(nn.Module):
    def __init__(self, vocab_size=len(vocab) + 1, embed_size=64, hidden_size=128):
        super(PHPGenerator, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)  # + hidden_size for attention
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Store encoder outputs for attention
        self.encoder_outputs = None
    
    def forward(self, input_seq, output_seq=None, max_len=50):
        # Encode input
        embedded = self.embedding(input_seq)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded)
        self.encoder_outputs = encoder_outputs  # Store for attention
        
        # If training, use teacher forcing with attention
        if output_seq is not None:
            decoder_input = self.embedding(output_seq)
            decoder_outputs = []
            
            for t in range(decoder_input.size(1)):
                decoder_embed = decoder_input[:, t:t+1]  # [batch, 1, embed_size]
                
                # Attention mechanism
                attn_weights = self._attention(hidden.squeeze(0), encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_size]
                
                # Combine decoder input with context
                decoder_input_combined = torch.cat((decoder_embed, context), dim=2)
                
                decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input_combined, (hidden, cell))
                decoder_outputs.append(decoder_output)
            
            output = torch.cat(decoder_outputs, dim=1)
            output = self.fc(output)
            return output
        
        # If inference, generate token by token with attention
        else:
            generated = []
            # Start with SOS token (0)
            current_input = torch.tensor([[0]], dtype=torch.long).to(input_seq.device)
            
            for _ in range(max_len):
                embedded = self.embedding(current_input)
                
                # Attention mechanism
                attn_weights = self._attention(hidden.squeeze(0), encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_size]
                
                # Combine decoder input with context
                decoder_input_combined = torch.cat((embedded, context), dim=2)
                
                decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input_combined, (hidden, cell))
                output = self.fc(decoder_output)
                _, next_token = torch.max(output, dim=-1)
                generated.append(next_token.item())
                current_input = next_token
            
            return generated
    
    def _attention(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        
        # Compute attention scores
        attn_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores.squeeze(2), dim=1)  # [batch, seq_len]
        
        return attn_weights

def generate_php_code(description, model, vocab, reverse_vocab):
    input_ids = tokenize(description)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        generated_ids = model(input_tensor, output_seq=None)
    
    # Decode generated tokens
    generated_tokens = [reverse_vocab.get(id, '') for id in generated_ids if id != 0]
    generated_code = ' '.join(generated_tokens)
    # Simple formatting
    generated_code = generated_code.replace(' <?php', '<?php').replace(' \n ', '\n').replace(' \n', '\n').replace(' ?>', '?>\n')
    
    return generated_code

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
        generated_code = generate_php_code(test_input, model, vocab, reverse_vocab)
        
        print(f"\n💭 Input: {test_input}")
        print(f"📝 Generated:\n{generated_code}")
    
    return model

# Save the model
if __name__ == "__main__":
    model = train_php_ai()
    torch.save(model.state_dict(), 'php_generator.pth')
    print("\n💾 Model saved as 'php_generator.pth'")
    print("💾 Vocab saved as 'vocab.pkl'")