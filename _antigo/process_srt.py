import re
import os

def process_srt_to_text_interactive():
    # Pedir o caminho do arquivo
    srt_file_path = input("Caminho do arquivo SRT: ").strip()
    
    # Validar se arquivo existe
    if not os.path.exists(srt_file_path):
        print(f"Erro: Arquivo '{srt_file_path}' não encontrado!")
        return
    
    # Definir saída (mesma pasta, sem extensão)
    output_file_path = srt_file_path.replace('.srt', '_processado.txt')
    
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    processed_text = []
    current_subtitle = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_subtitle:
                processed_text.append(' '.join(current_subtitle))
                current_subtitle = []
            continue
        
        if re.match(r'^\d+$', line):
            continue
        
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
            continue
        
        current_subtitle.append(line)
    
    if current_subtitle:
        processed_text.append(' '.join(current_subtitle))
    
    final_text = ' '.join(processed_text)
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(final_text)
    
    print(f"\n✅ Arquivo salvo em: {output_file_path}")
    print(f"\nPrévia:\n{final_text[:300]}...\n")

# Executar
process_srt_to_text_interactive()