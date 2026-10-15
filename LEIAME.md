# Transcritor

Transforma **vídeo, áudio ou legenda SRT** em um **texto organizado por pessoa**, com frases
completas e em ordem — pronto para anexar no ChatGPT, Claude ou outra IA. Tudo roda **no seu
computador** (sem internet, sem conta, sem limite de minutos) depois de instalado.

## Como usar

1. Dê dois cliques no atalho **Transcritor** da Área de Trabalho
   (ou arraste um ou vários arquivos para cima dele).
2. Escolha os arquivos na janela do Windows — vídeo, áudio ou legenda `.srt`/`.vtt`.
   O programa descobre sozinho o que é cada um.
3. Escolha onde salvar o texto (`.txt`). Se for vídeo, ele também pergunta onde salvar o
   **áudio extraído** (cancele essa janela se não quiser o áudio).
4. Responda quantas pessoas falam (ou só Enter para ele descobrir sozinho).
5. Aguarde: a barra mostra o andamento e o tempo que falta. Pode continuar usando o PC.
6. **Quem é quem:** para cada voz encontrada aparecem frases de exemplo com o momento da
   gravação, ex.: `[03:15] "Houve uma tentativa de estorno..."`.
   - digite **1**, **2** ou **3** para **ouvir** aquele trecho;
   - digite **M** para ver mais exemplos;
   - digite o **nome** da pessoa (ou Enter para manter "Pessoa A").
   - Deu o **mesmo nome** a duas vozes? Elas viram uma só pessoa no texto.
7. No final aparece o resumo. Opções: **A** abre o TXT, **P** abre a pasta,
   **N** transcreve outros arquivos, **Enter** sai. A janela nunca fecha sozinha.

### Exemplo do texto gerado

```
TRANSCRIÇÃO DE VÍDEO
Origem: reuniao_banco.mp4
Duração: 52 min 13 s
Processado em: 14/10/2026 às 16:20
Participantes: Fulano (eu), Atendente, Gerente
Observações:
- Transcrição automática, em ordem cronológica; pode conter pequenos erros.
- "[Não identificado]" = trecho em que não foi possível saber quem falou.
================================================================

Fulano: Boa tarde. Houve uma tentativa de estorno, porém não houve efetivação.

Atendente: Senhor, deixa eu verificar no sistema.
```

## Reconhecer a sua voz

Na primeira vez, o programa pergunta **qual das vozes é a sua**. A partir daí ele guarda a sua
"impressão vocal" (só números, nunca o áudio) e, nas próximas gravações, a sua voz já vem com
o seu nome — basta confirmar com Enter. Quanto mais você usa, melhor ele reconhece.

## Dicas para o melhor resultado

- **Feche Chrome, ChatGPT, Vibe e outros programas pesados** antes de transcrever: o
  computador tem pouca memória livre. O programa avisa se a memória estiver baixa.
  Se outro programa pesado continuar aberto, o Transcritor (que roda com prioridade baixa
  para não travar o PC) fica bem mais lento — feche-o ou mude `prioridade_baixa = false`.
- **Informe o número de pessoas** quando souber: a separação das vozes fica mais precisa.
- Gravações com menos ruído e cada pessoa perto do microfone dão resultados melhores.
- Tempo: o processamento leva **menos que a duração da gravação**. Medido neste computador:
  um vídeo de 58 minutos levou 43 minutos (com outro programa pesado aberto ao mesmo tempo),
  usando no máximo 0,9 GB de memória. Se for interrompido (Ctrl+C, falta de energia), abra
  de novo e escolha o mesmo arquivo: ele **continua de onde parou**.
- Abrir de novo um arquivo já transcrito é instantâneo: dá para só renomear as vozes.

## Configurações

Abra `config.toml` com o Bloco de Notas. Cada opção tem explicação. As mais úteis:

| Opção | O que faz |
|---|---|
| `mostrar_horarios` | coloca `[12:34]` no início de cada fala (útil para citar trechos) |
| `uma_frase_por_linha` | cada frase numa linha, com o nome da pessoa |
| `separar_vozes` | `false` = texto corrido, sem identificar quem fala (mais rápido) |
| `sensibilidade_separacao` | aumente se juntar pessoas diferentes; diminua se dividir uma pessoa |
| `salvar_audio_extraido` | salvar ou não o áudio dos vídeos |
| `memorizar_outras_vozes` | além da sua, lembrar das vozes de quem você nomear |

## Onde ficam as coisas

| Pasta/arquivo | Conteúdo |
|---|---|
| `Transcritor.bat` | o programa (o atalho da Área de Trabalho aponta para ele) |
| `modelos\` | modelos de IA baixados uma única vez (~700 MB) |
| `dados\cache\` | trabalho já feito por arquivo (apagado sozinho depois de 30 dias sem uso) |
| `dados\vozes.json` | impressões vocais conhecidas (a sua e, se configurado, outras) |
| `logs\` | um registro por execução, com o passo a passo e os erros (sem o texto das conversas) |
| `_antigo\process_srt.py` | o script antigo, guardado como cópia de segurança |

## Problemas comuns

- **"Memória livre baixa"**: é só um aviso — vai funcionar, apenas mais devagar.
- **"Memória quase esgotada"**: aí há risco de falhar. Feche outros programas (Chrome,
  ChatGPT, Vibe) e digite **V** para verificar de novo.
- **"Não consegui salvar ... está aberto em outro programa"**: feche o arquivo no Word e
  digite **T**, ou **O** para salvar com outro nome.
- **"Não foi possível abrir como áudio ou vídeo"**: o arquivo está danificado ou não é mídia.
- **Algo estranho depois de uma atualização**: apague a pasta `.venv` e abra o Transcritor;
  ele se reinstala sozinho (precisa de internet).
- Qualquer erro fica explicado na tela e detalhado no log mais recente da pasta `logs`.

## Privacidade

Tudo é processado **no seu computador**. Nenhum áudio ou texto é enviado para a internet
(a internet só é usada uma vez, para baixar os programas e os modelos).

## Como funciona (técnico)

- **Transcrição:** NVIDIA Parakeet TDT 0.6B v3 (int8), via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
  100% na CPU. Erro em português ≈ 5% no teste FLEURS (o Whisper large-v3-turbo tem ≈ 4%, mas
  usa quase o dobro de memória). Pico de memória medido neste PC: ~0,9 GB.
- **Detecção de fala:** Silero VAD; o áudio é dividido em pedaços de até 20 s nas pausas, para
  usar pouca memória em gravações de qualquer duração.
- **Separação de vozes:** segmentação pyannote 3.0 + impressão vocal NVIDIA TitaNet small.
  Escolhido por testes com vozes humanas reais (`ferramentas\avaliar_separacao.py`):
  93% da fala atribuída à pessoa certa numa conversa de 3 pessoas reais e 100% no diálogo
  em português, encontrando sozinho o número de pessoas. O modelo WeSpeaker, testado antes,
  quase não distinguia pessoas diferentes e foi descartado.
- **Frases:** pontuação do modelo + regras para abreviações em português (Sr., Dra., R$ 1.500,00).

## Alternativas prontas (pesquisa de outubro/2026)

| Onde | Ferramenta | Grátis | Separa vozes | Observações |
|---|---|---|---|---|
| Navegador | TurboScribe | 3 arquivos/dia, até 30 min cada | Sim | qualidade Whisper large |
| Navegador | Speechmatics | US$ 100 de crédito por 12 meses | Sim | excelente em português |
| Navegador | AssemblyAI | US$ 50 de crédito | Sim | português com separação de vozes |
| Local | Vibe | sempre | Sim | aplicativo com janela; pesado com outros programas abertos |

## Para desenvolvedores

```
.venv\Scripts\python -m pytest                 testes rápidos
.venv\Scripts\python -m pytest -m lento        testes com os modelos de IA
.venv\Scripts\python -m ruff check .           revisão de estilo
.venv\Scripts\python -m transcritor arquivo.mp4 --sem-perguntas --saida pasta --pessoas 2
py -3.12 instalar.py --dev                     (re)instala tudo, com as ferramentas de teste
```

`ferramentas\gerar_dialogo_teste.py` cria um diálogo de teste com as vozes do Windows;
`ferramentas\avaliar_separacao.py` compara modelos de separação de vozes com gravações reais;
`ferramentas\gerar_icone.py` desenha o ícone.
