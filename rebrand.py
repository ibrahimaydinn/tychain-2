import os
import glob

def rebrand(directory):
    for root, dirs, files in os.walk(directory):
        if '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith(('.py', '.html', '.md', '.sh', '.json', '.txt')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content.replace('Tychain', 'Tychain').replace('tychain', 'tychain')
                    
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Rebranded {path}")
                except Exception as e:
                    print(f"Error on {path}: {e}")

if __name__ == '__main__':
    rebrand('.')
    
    # Also rename tychain.db to tychain.db
    if os.path.exists('tychain.db'):
        os.rename('tychain.db', 'tychain.db')
        print("Renamed tychain.db to tychain.db")
