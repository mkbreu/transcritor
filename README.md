# Transcritor

Transforma **vídeo, áudio ou legenda SRT** em um **texto organizado por pessoa**, com frases
completas e em ordem — pronto para anexar no ChatGPT, Claude ou outra IA. Tudo roda **no seu
computador** (sem internet, sem conta, sem limite de minutos) depois de instalado.

Gratuito para uso **não comercial** — licença [PolyForm Noncommercial 1.0.0](LICENSE).

## Instalação

Precisa de **Windows 10 ou 11**, **Python 3.12** ([python.org](https://www.python.org/downloads/))
e ~1,5 GB livres no disco. Funciona em computadores modestos (testado num i5 de 2013 com 8 GB).

1. Baixe este projeto (botão **Code → Download ZIP**) e descompacte numa pasta.
2. Dê dois cliques em **`Transcritor.bat`**. Na primeira vez ele prepara tudo sozinho (precisa
   de internet): cria um ambiente isolado, baixa os pacotes e os modelos de IA (~690 MB) e cria
   o atalho **Transcritor** na Área de Trabalho.

## Como usar

1. Dê dois cliques no atalho **Transcritor** da Área de Trabalho
   (ou arraste um ou vários arquivos para cima dele).
2. Escolha os arquivos na janela do Windows — vídeo, áudio ou legenda `.srt`/`.vtt`.
   O programa descobre sozinho o que é cada um.
3. Escolha onde salvar o texto (`.txt`). Se for vídeo, ele também pergunta onde salvar o
   **áudio extraído** (cancele essa janela se não quiser o áudio).
4. Responda quantas pessoas falam (ou só Enter para ele descobrir sozinho). O número é só
   uma **pista**: o programa sempre confere pela voz quantas pessoas falam de fato. Se achar
   menos do que você disse, ele refaz a separação procurando o seu número; se achar mais,
   mostra todas para você conferir.
5. Aguarde: a barra mostra o andamento e o tempo que falta. Pode continuar usando o PC.
6. **Quem é quem:** para cada voz encontrada aparecem frases de exemplo com o momento da
   gravação, ex.: `[03:15] "Houve uma tentativa de estorno..."`.
   - digite **1**, **2** ou **3** para **ouvir** aquele trecho;
   - digite **M** para ver mais exemplos;
   - digite o **nome** da pessoa (ou Enter para manter "Pessoa A");
   - para a **sua** voz, escreva **(eu)** depois do nome — ex.: `Maykon (eu)` — ou só `eu`.
     Pode marcar quantas vozes forem suas: todas viram você no texto.
   - Deu o **mesmo nome** a duas vozes? Elas viram uma só pessoa no texto.
   - O programa prefere **sobrar** uma voz a juntar duas pessoas diferentes (isso não teria
     conserto). Se aparecer a mesma pessoa duas vezes, é só dar o mesmo nome.
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
- [início – fim] = trecho da gravação em que aquela fala acontece.
================================================================

[00:00 – 00:10] Fulano: Boa tarde. Houve uma tentativa de estorno, porém não houve efetivação.

[00:11 – 00:16] Atendente: Senhor, deixa eu verificar no sistema.
```

Cada parágrafo é uma fala completa de uma pessoa; quando outra pessoa começa a falar, começa
um novo parágrafo (se a fala foi cortada no meio, ela termina com "…").

## Reconhecer a sua voz

Marque a sua voz com **(eu)** ao dar os nomes (ou responda, no final, quais letras são suas —
pode ser mais de uma, ex.: `A C`). O programa guarda a sua "impressão vocal" (só números,
nunca o áudio) e, nas próximas gravações, a sua voz já vem com o seu nome — basta confirmar
com Enter. Quanto mais você usa, melhor ele reconhece.

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
| `mostrar_horarios` | `[início – fim]` de cada parágrafo (ligado); `false` = texto sem horários |
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
- **Separação de vozes:** segmentação **Reverb diarization v1** (Rev) + impressão vocal
  **NVIDIA TitaNet small**, seguidas de um **refino**: vozes com menos de 2 s de fala (tosse,
  ruído) são absorvidas, e grupos cujas impressões vocais médias são parecidas (≥ 0,60) viram
  a mesma pessoa. Sem isso, uma gravação real de 47 min de uma pessoa só dava 12 "vozes".
  Tudo foi escolhido e calibrado com vozes humanas reais (`ferramentas\avaliar_separacao.py`):
  a Reverb partiu muito menos a mesma pessoa que a segmentação pyannote 3.0 (5 grupos em vez
  de 12) e deixou pessoas diferentes mais distintas (semelhança máxima 0,44 em vez de 0,57),
  o que dá uma margem de 0,16 contra o erro sem conserto (misturar duas pessoas). TitaNet
  small venceu os modelos maiores testados (TitaNet large, ERes2NetV2, WeSpeaker, CAM++).
- **Frases:** pontuação do modelo + regras para abreviações em português (Sr., Dra., R$ 1.500,00).

## Licenças dos componentes

O Transcritor baixa os modelos de IA dos lançamentos oficiais do projeto sherpa-onnx; cada um
tem a sua licença:

| Componente | Licença |
|---|---|
| sherpa-onnx (motor) | Apache 2.0 |
| NVIDIA Parakeet TDT 0.6B v3 (transcrição) | CC BY 4.0 |
| NVIDIA TitaNet small (impressão vocal) | CC BY 4.0 |
| Silero VAD (detecção de fala) | MIT |
| **Reverb diarization v1 (separação de vozes)** | **Rev Model Non-Production License** |
| PyAV / FFmpeg, numpy, rich | BSD / LGPL, BSD, MIT |

**Atenção — Reverb:** "Licensed by Rev under the Rev Model Non-Production License". Pode ser
usada para fins **pessoais, de pesquisa ou avaliação**; não pode ser usada em atividade
comercial (inclusive como serviço). Para uso comercial é preciso licença da Rev
(licensing@rev.com).

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

## Licença

O código do Transcritor está sob a [PolyForm Noncommercial License 1.0.0](LICENSE): qualquer
pessoa pode usar, copiar, modificar e compartilhar **de graça, para fins não comerciais**
(uso pessoal, estudo, pesquisa, projetos sociais, entidades sem fins lucrativos). Uso comercial
não é permitido. Os modelos de IA baixados na instalação têm as licenças próprias listadas em
[Licenças dos componentes](#licenças-dos-componentes).
