# php_ai.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import pickle
import random
import json
from collections import defaultdict

print("🚀 Building Advanced PHP Code Assistant AI with Conversational Capabilities...")

# Expanded dataset with PHP examples and conversational responses
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
     "<?php\n$password = 'mypassword';\n$hash = '$2y$10$...';\nif (password_verify($password, $hash)) {\n    echo 'Valid password';\n}\n?>"],

    ["create a class",
     "<?php\nclass User {\n    public $name;\n    public $email;\n    \n    public function __construct($name, $email) {\n        $this->name = $name;\n        $this->email = $email;\n    }\n    \n    public function displayInfo() {\n        echo 'Name: ' . $this->name . ', Email: ' . $this->email;\n    }\n}\n?>"],

    ["create session",
     "<?php\nsession_start();\n$_SESSION['user_id'] = 123;\n$_SESSION['username'] = 'john_doe';\necho 'Session created';\n?>"],

    ["destroy session",
     "<?php\nsession_start();\nsession_destroy();\necho 'Session destroyed';\n?>"],

    ["redirect to another page",
     "<?php\nheader('Location: dashboard.php');\nexit();\n?>"],

    ["set cookie",
     "<?php\nsetcookie('user', 'John', time() + 86400, '/');\necho 'Cookie set';\n?>"],

    ["get cookie",
     "<?php\nif(isset($_COOKIE['user'])) {\n    echo 'User: ' . $_COOKIE['user'];\n}\n?>"],

    ["delete cookie",
     "<?php\nsetcookie('user', '', time() - 3600, '/');\necho 'Cookie deleted';\n?>"],

    ["create a function",
     "<?php\nfunction addNumbers($a, $b) {\n    return $a + $b;\n}\necho addNumbers(5, 3); // Output: 8\n?>"],

    ["create an array",
     "<?php\n$fruits = array('apple', 'banana', 'orange');\nprint_r($fruits);\n?>"],

    ["loop through array",
     "<?php\n$colors = ['red', 'green', 'blue'];\nforeach($colors as $color) {\n    echo $color . ' ';\n}\n?>"],

    ["if else statement",
     "<?php\n$age = 25;\nif($age >= 18) {\n    echo 'Adult';\n} else {\n    echo 'Minor';\n}\n?>"],

    ["switch statement",
     "<?php\n$day = 'Monday';\nswitch($day) {\n    case 'Monday':\n        echo 'Start of work week';\n        break;\n    case 'Friday':\n        echo 'Weekend is coming';\n        break;\n    default:\n        echo 'Regular day';\n}\n?>"],

    ["for loop",
     "<?php\nfor($i = 1; $i <= 5; $i++) {\n    echo $i . ' ';\n}\n?>"],

    ["while loop",
     "<?php\n$i = 1;\nwhile($i <= 5) {\n    echo $i . ' ';\n    $i++;\n}\n?>"],

    ["do while loop",
     "<?php\n$i = 1;\ndo {\n    echo $i . ' ';\n    $i++;\n} while($i <= 5);\n?>"]
]

# Conversational responses dataset
conversational_examples = [
    # Greetings
    ["hi", "Hello! How can I help you with PHP today?"],
    ["hello", "Hi there! I'm your PHP assistant. What would you like to learn or create?"],
    ["hey", "Hey! Ready to write some PHP code?"],
    ["good morning", "Good morning! Looking for PHP assistance?"],
    ["good afternoon", "Good afternoon! How can I help with PHP?"],
    ["good evening", "Good evening! Need help with PHP programming?"],
    
    # Basic questions
    ["how are you", "I'm doing great! Ready to help you with PHP programming."],
    ["what is your name", "I'm PHP Code Assistant AI, designed to help you write and understand PHP code."],
    ["who created you", "I was created by developers to assist with PHP programming tasks."],
    ["what can you do", "I can generate PHP code, answer PHP-related questions, explain concepts, and help you learn PHP programming."],
    
    # PHP-related questions
    ["what is php", "PHP is a popular server-side scripting language designed for web development. It's used to create dynamic web pages and applications. PHP stands for Hypertext Preprocessor."],
    
    ["php advantages", "PHP advantages include:\n1. Open source and free\n2. Easy to learn and use\n3. Cross-platform compatibility\n4. Large community support\n5. Extensive documentation\n6. Built-in database connectivity\n7. Fast execution\n8. Secure features available"],
    
    ["php vs python", "PHP vs Python for web development:\nPHP: Specifically designed for web, excellent for server-side scripting, integrates well with HTML.\nPython: General purpose, great for data science and AI, uses frameworks like Django/Flask for web.\nPHP is more web-focused, while Python is more versatile."],
    
    ["php frameworks", "Popular PHP frameworks include:\n1. Laravel - Most popular, elegant syntax\n2. Symfony - Enterprise-level, reusable components\n3. CodeIgniter - Lightweight and fast\n4. Yii - High performance, security-focused\n5. CakePHP - Rapid development\n6. Zend Framework - Professional grade"],
    
    ["how to learn php", "To learn PHP effectively:\n1. Start with basic syntax\n2. Practice with simple scripts\n3. Learn about forms and user input\n4. Study database connectivity (MySQL)\n5. Understand sessions and cookies\n6. Learn object-oriented PHP\n7. Explore PHP frameworks\n8. Build real projects"],
    
    ["php security best practices", "PHP security best practices:\n1. Validate and sanitize all user inputs\n2. Use prepared statements for SQL queries\n3. Hash passwords with password_hash()\n4. Use HTTPS for sensitive data\n5. Implement CSRF protection\n6. Keep PHP updated\n7. Use secure session handling\n8. Limit file uploads\n9. Escape output data"],
    
    ["php error handling", "PHP error handling methods:\n1. try-catch blocks for exceptions\n2. set_error_handler() for custom error handling\n3. error_reporting() to control error levels\n4. @ operator to suppress errors (use sparingly)\n5. Log errors for debugging\n6. Display user-friendly error messages"],
    
    ["php oop concepts", "PHP Object-Oriented Programming concepts:\n1. Classes and Objects\n2. Inheritance (extends)\n3. Encapsulation (public/private/protected)\n4. Polymorphism\n5. Abstraction (abstract classes)\n6. Interfaces\n7. Traits\n8. Constructors and Destructors"],
    
    ["php database connection", "To connect PHP with databases:\n1. MySQLi - Improved MySQL extension\n2. PDO - Database abstraction layer\n3. Use prepared statements to prevent SQL injection\n4. Handle connection errors gracefully\n5. Close connections when done"],
    
    ["php file handling", "PHP file handling functions:\n1. fopen() - Open files\n2. fread() - Read files\n3. fwrite() - Write to files\n4. fclose() - Close files\n5. file_get_contents() - Read entire file\n6. file_put_contents() - Write entire file\n7. unlink() - Delete files"],
    
    # Technical explanations
    ["explain php mvc", "PHP MVC (Model-View-Controller) is an architectural pattern:\n\nModel: Handles data logic and database interactions\nView: Presents data to users (HTML templates)\nController: Processes user requests and coordinates between Model and View\n\nBenefits: Separation of concerns, code reusability, easier maintenance."],
    
    ["what is composer", "Composer is a dependency manager for PHP. It allows you to:\n1. Declare libraries your project depends on\n2. Install and update dependencies\n3. Manage autoloading\n4. Share packages on Packagist\n5. Handle version constraints"],
    
    ["what is laravel", "Laravel is a popular PHP framework that provides:\n1. Elegant syntax and expressive code\n2. Built-in authentication and authorization\n3. Eloquent ORM for database operations\n4. Blade templating engine\n5. Artisan command-line tool\n6. Queue system for background jobs\n7. API development support"],
    
    ["php version differences", "Major PHP version differences:\nPHP 5.x: Older version, less security\nPHP 7.x: 2x faster, better error handling, return type declarations\nPHP 8.x: JIT compilation, named arguments, union types, match expression, attributes\nAlways use the latest stable version for better performance and security."],
    
    ["php debugging tools", "PHP debugging tools include:\n1. Xdebug - Powerful debugger and profiler\n2. PHP Error - Better error reporting\n3. Whoops - Pretty error interface\n4. Kint - Replacement for var_dump()\n5. Laravel Debugbar (for Laravel)\n6. Blackfire.io - Profiling tool"],
    
    # Code explanation requests
    ["explain this code", "I'd be happy to explain PHP code for you. Please share the code snippet you'd like me to explain."],
    
    ["how does this work", "To help you understand how PHP code works, please provide the code you're curious about, and I'll break it down step by step."],
    
    ["why use php", "You should use PHP because:\n1. Perfect for web development\n2. Easy integration with HTML\n3. Great database support\n4. Massive community and resources\n5. Continuous improvements\n6. Cost-effective (open source)\n7. Works with all major servers\n8. Excellent for CMS like WordPress"],
    
    # Project-related
    ["php project ideas", "PHP project ideas for practice:\n1. Blog/CMS system\n2. E-commerce website\n3. Social media platform\n4. Task management app\n5. Hotel booking system\n6. Online quiz platform\n7. File sharing system\n8. Portfolio website\n9. API for mobile app\n10. Learning management system"],
    
    ["best php practices", "Best PHP coding practices:\n1. Follow PSR standards\n2. Use meaningful variable names\n3. Comment your code\n4. Separate logic from presentation\n5. Use functions and classes\n6. Handle errors gracefully\n7. Optimize database queries\n8. Secure your applications\n9. Test your code\n10. Keep code DRY (Don't Repeat Yourself)"],
    
    # Troubleshooting
    ["php error", "To help with PHP errors, please share:\n1. The error message\n2. Code causing the error\n3. PHP version\nCommon errors include syntax errors, undefined variables, and database connection issues."],
    
    ["php not working", "If PHP is not working:\n1. Check if PHP is installed (php -v)\n2. Ensure web server is running\n3. Check file extension (.php)\n4. Look for syntax errors\n5. Verify file permissions\n6. Check error logs"],
    
    # Thank you and farewell
    ["thank you", "You're welcome! Happy coding! Let me know if you need more PHP help."],
    ["thanks", "You're welcome! Feel free to ask if you have more PHP questions."],
    ["bye", "Goodbye! Come back anytime for PHP assistance."],
    ["goodbye", "Goodbye! Keep learning PHP!"],
    ["see you", "See you! Remember to practice your PHP skills regularly."],
    
    # Encouragement
    ["i'm stuck", "Don't worry! PHP has a learning curve. Try breaking down the problem into smaller parts, search for similar solutions, or ask for help in PHP communities. You can do it!"],
    
    ["beginner php", "For PHP beginners:\n1. Start with basic echo statements\n2. Learn variables and data types\n3. Practice with forms\n4. Connect to a database\n5. Build a simple CRUD application\n6. Take it one step at a time"],
    
    # Additional conversational variations
    ["help me", "I'm here to help! What PHP topic or code do you need assistance with?"],
    ["can you help", "Absolutely! I specialize in PHP. What do you need help with?"],
    ["need help", "Sure! Tell me what PHP problem you're facing or what you want to create."],
    
    ["what's new in php", "New features in recent PHP versions:\nPHP 8.1: Enums, readonly properties, fibers\nPHP 8.2: Readonly classes, disjunctive normal form types, random extension\nPHP 8.3: Typed class constants, dynamic class constant fetch, json_validate()"],
    
    ["php performance tips", "PHP performance optimization tips:\n1. Use OPcache for bytecode caching\n2. Optimize database queries with indexes\n3. Minimize file I/O operations\n4. Use efficient data structures\n5. Enable gzip compression\n6. Cache frequently used data\n7. Use CDN for static assets\n8. Profile code to find bottlenecks"],
]

# Combine all examples
all_examples = php_examples + conversational_examples

# Add more variations for robustness
additional_variations = []
for desc, code in all_examples:
    # Add lowercase variations
    additional_variations.append([desc.lower(), code])
    # Add uppercase variations for first word
    additional_variations.append([desc.capitalize(), code])
    # Add question variations
    if not desc.endswith('?'):
        additional_variations.append([desc + "?", code])
        additional_variations.append(["how to " + desc + "?", code])

all_examples.extend(additional_variations)

# Build global vocabulary with special tokens
all_texts = [desc + ' ' + code for desc, code in all_examples]
all_tokens = set()

# Add special tokens
special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
for token in special_tokens:
    all_tokens.add(token)

for text in all_texts:
    # More sophisticated tokenization
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    all_tokens.update(tokens)

vocab = {token: idx + 1 for idx, token in enumerate(sorted(all_tokens))}  # 0 for padding/unknown
reverse_vocab = {idx: token for token, idx in vocab.items()}

# Add special token mappings
for i, token in enumerate(special_tokens, start=1):
    vocab[token] = len(vocab) + 1
    reverse_vocab[len(vocab)] = token

# Save vocab
with open('vocab.pkl', 'wb') as f:
    pickle.dump((vocab, reverse_vocab), f)

print(f"📊 Vocabulary size: {len(vocab)} tokens")
print(f"📚 Total training examples: {len(all_examples)}")

# Enhanced tokenizer
def tokenize(text, max_len=100):
    # Convert text to lowercase and split
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    
    # Add SOS token
    token_ids = [vocab.get('<SOS>', 0)]
    
    # Convert to IDs using global vocab
    for token in tokens:
        token_ids.append(vocab.get(token, vocab.get('<UNK>', 0)))
    
    # Add EOS token
    token_ids.append(vocab.get('<EOS>', 0))
    
    # Pad/truncate to fixed length
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len-1] + [vocab.get('<EOS>', 0)]
    else:
        token_ids = token_ids + [vocab.get('<PAD>', 0)] * (max_len - len(token_ids))
    
    return token_ids

def detokenize(token_ids, reverse_vocab):
    tokens = []
    for token_id in token_ids:
        if token_id == 0:  # PAD
            continue
        token = reverse_vocab.get(token_id, '')
        if token in ['<SOS>', '<EOS>', '<PAD>']:
            continue
        tokens.append(token)
    return ' '.join(tokens)

class EnhancedPHPAIDataset(Dataset):
    def __init__(self, examples, max_len=100):
        self.examples = examples
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        desc, code = self.examples[idx]
        
        # Tokenize input (description) and output (code)
        input_ids = tokenize(desc, self.max_len)
        output_ids = tokenize(code, self.max_len)
        
        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'output': torch.tensor(output_ids, dtype=torch.long)
        }

# Enhanced model with better architecture
class EnhancedPHPGenerator(nn.Module):
    def __init__(self, vocab_size=max(vocab.values()) + 1, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super(EnhancedPHPGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Bidirectional Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            embed_size + hidden_size * 2,  # *2 for bidirectional
            hidden_size * 2,  # Double for bidirectional context
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # *4: decoder_hidden + encoder_output
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # *4: decoder_hidden + context
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 4)
    
    def forward(self, input_seq, output_seq=None, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        
        # Encode input
        embedded_input = self.embedding_dropout(self.embedding(input_seq))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_input)
        
        # Prepare decoder initial states (average bidirectional states)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.mean(hidden, dim=1)  # Average bidirectional
        hidden = hidden.repeat(1, 1, 2)  # Double for decoder
        
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = torch.mean(cell, dim=1)
        cell = cell.repeat(1, 1, 2)
        
        if output_seq is not None:  # Training mode
            seq_len = output_seq.size(1)
            decoder_input = self.embedding_dropout(self.embedding(output_seq))
            
            outputs = []
            
            for t in range(seq_len - 1):
                # Attention mechanism
                attn_weights = self.compute_attention(hidden[-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                
                # Combine decoder input with context
                decoder_input_t = decoder_input[:, t:t+1]
                combined = torch.cat([decoder_input_t, context], dim=-1)
                
                # Decoder step
                decoder_output, (hidden, cell) = self.decoder_lstm(combined, (hidden, cell))
                
                # Final output
                output = torch.cat([decoder_output, context], dim=-1)
                output = self.layer_norm(output)
                output = self.fc(output)
                
                outputs.append(output)
            
            outputs = torch.cat(outputs, dim=1)
            return outputs
        
        else:  # Inference mode
            max_len = 100
            generated = []
            
            # Start with SOS token
            decoder_input = torch.tensor([[vocab.get('<SOS>', 0)]], dtype=torch.long).to(input_seq.device)
            
            for _ in range(max_len):
                embedded = self.embedding_dropout(self.embedding(decoder_input))
                
                # Attention mechanism
                attn_weights = self.compute_attention(hidden[-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                
                # Combine decoder input with context
                combined = torch.cat([embedded, context], dim=-1)
                
                # Decoder step
                decoder_output, (hidden, cell) = self.decoder_lstm(combined, (hidden, cell))
                
                # Final output
                output = torch.cat([decoder_output, context], dim=-1)
                output = self.layer_norm(output)
                output = self.fc(output)
                
                # Get next token
                _, next_token = torch.max(output, dim=-1)
                generated.append(next_token.item())
                
                # Stop if EOS token
                if next_token.item() == vocab.get('<EOS>', 0):
                    break
                
                decoder_input = next_token
            
            return generated
    
    def compute_attention(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden_size*2]
        # encoder_outputs: [batch, seq_len, hidden_size*2]
        
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Expand decoder hidden for attention computation
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute attention scores
        combined = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=-1)
        attn_scores = self.attention(combined).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        return attn_weights

def generate_response(user_input, model, vocab, reverse_vocab, temperature=0.7):
    """Generate response with temperature sampling for more natural output"""
    model.eval()
    
    input_ids = tokenize(user_input)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        generated_ids = model(input_tensor, output_seq=None)
    
    # Decode with temperature sampling
    generated_text = detokenize(generated_ids, reverse_vocab)
    
    # Post-process the output
    generated_text = post_process_output(generated_text)
    
    return generated_text

def post_process_output(text):
    """Clean up and format the generated text"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix PHP tags
    text = text.replace(' <?php', '<?php').replace(' ?>', '?>')
    
    # Format code blocks
    if '<?php' in text:
        # Add newlines after semicolons and braces for readability
        text = text.replace(';', ';\n')
        text = text.replace('{', '{\n')
        text = text.replace('}', '\n}')
        
        # Clean up extra newlines
        text = re.sub(r'\n\s*\n', '\n', text)
    
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text.strip()

def classify_input_type(user_input):
    """Classify if the input is a code request or conversational"""
    user_input_lower = user_input.lower()
    
    # Keywords indicating code generation requests
    code_keywords = [
        'create', 'make', 'generate', 'write', 'code for', 'php for', 
        'how to', 'connect', 'validate', 'read', 'write', 'upload',
        'send', 'get', 'set', 'delete', 'update', 'insert', 'select',
        'function', 'class', 'form', 'database', 'query', 'api'
    ]
    
    # Keywords indicating conversational requests
    conversational_keywords = [
        'hi', 'hello', 'hey', 'how are', 'what is', 'explain',
        'help', 'thank', 'thanks', 'bye', 'goodbye', 'why',
        'when', 'where', 'who', 'which', 'tell me about'
    ]
    
    # Check for code keywords
    for keyword in code_keywords:
        if keyword in user_input_lower:
            return 'code'
    
    # Check for conversational keywords
    for keyword in conversational_keywords:
        if keyword in user_input_lower:
            return 'conversation'
    
    # Default to conversation for general queries
    return 'conversation'

def train_enhanced_php_ai():
    print("📚 Training Enhanced PHP Code Generator with Conversation...")
    
    # Create dataset
    dataset = EnhancedPHPAIDataset(all_examples, max_len=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = EnhancedPHPGenerator()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get('<PAD>', 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_seq = batch['input']
            target_seq = batch['output']
            
            # Forward pass
            output = model(input_seq, target_seq[:, :-1])
            
            # Calculate loss
            target = target_seq[:, 1:1+output.size(1)]
            loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'vocab': vocab,
                    'reverse_vocab': reverse_vocab
                }, 'enhanced_php_generator_best.pth')
                print(f"💾 Saved best model with loss: {best_loss:.4f}")
    
    # Final save
    torch.save(model.state_dict(), 'enhanced_php_generator.pth')
    print(f"\n✅ Training complete! Final loss: {avg_loss:.4f}")
    
    return model

def interactive_chat(model, vocab, reverse_vocab):
    """Interactive chat interface for the PHP assistant"""
    print("\n" + "="*60)
    print("🤖 PHP Code Assistant AI - Interactive Mode")
    print("="*60)
    print("Type your questions or code requests.")
    print("Examples:")
    print("  - 'How to connect to MySQL database?'")
    print("  - 'What is PHP?'")
    print("  - 'Create a login form'")
    print("  - 'Explain OOP in PHP'")
    print("  - Type 'quit' or 'exit' to end")
    print("="*60)
    
    model.eval()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n🤖 Assistant: Goodbye! Happy coding!")
                break
            
            if not user_input:
                continue
            
            # Classify input type
            input_type = classify_input_type(user_input)
            
            # Generate response
            response = generate_response(user_input, model, vocab, reverse_vocab)
            
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            print("🤖 Assistant: I encountered an error. Please try again.")

# Test function
def test_model(model, vocab, reverse_vocab):
    """Test the model with various inputs"""
    print("\n🧪 Testing Enhanced PHP Generator:")
    
    test_cases = [
        "hi",
        "hello there",
        "how are you?",
        "what is php?",
        "connect to mysql database",
        "validate email address",
        "create a login form",
        "explain php mvc",
        "how to learn php",
        "php security best practices",
        "create a class in php",
        "send email with php",
        "thank you",
        "bye"
    ]
    
    for test_input in test_cases:
        response = generate_response(test_input, model, vocab, reverse_vocab)
        
        print(f"\n💭 Input: {test_input}")
        print(f"🤖 Response:\n{response}")
        print("-" * 50)

# Main execution
if __name__ == "__main__":
    print("🚀 Starting PHP AI Assistant Setup...")
    
    # Train the model
    model = train_enhanced_php_ai()
    
    # Load the best model
    checkpoint = torch.load('enhanced_php_generator_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    reverse_vocab = checkpoint['reverse_vocab']
    
    print("\n✅ Model loaded successfully!")
    
    # Test the model
    test_model(model, vocab, reverse_vocab)
    
    # Start interactive chat
    interactive_chat(model, vocab, reverse_vocab)
    
    print("\n💾 Model saved as 'enhanced_php_generator.pth'")
    print("💾 Best model saved as 'enhanced_php_generator_best.pth'")
    print("💾 Vocab saved as 'vocab.pkl'")
    print("\n🎉 PHP AI Assistant is ready to help!")