# pico-llm

## Task 1: 
1. run ```pip install torch datasets tiktoken``` to install tiktoken
2. download ```3seqs.txt```, and save it in the same folder where your code is.
3. go to the section titled Models, find the models = {...} dictionary and uncomment the ones you want to run.
4. run the code in the commander using this command ```python pico-llm_project.py --block_size 32 --tinystories_weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4"```
5. output should be somthing like this
```
'Requested device 'cuda:0' but CUDA not available. Falling back to CPU.
Using device: cpu, block_size=32, kgram_k=3, chunk_size=1, embed_size=1024
TinyStories weight=0 => skipping TinyStories.
Vocab size: 50257
Reading custom text file: 3seqs.txt    
Custom input files: 3 sequences loaded.

=== Training model: lstm_seq ===

[lstm_seq] Generating sample text (greedy) at epoch=1, step=1...
 Greedy Sample: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn
 Annotated: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn

[lstm_seq] Generating sample text (top-p=0.95) at epoch=1, step=1...
 Top-p (p=0.95) Sample: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn
 Annotated: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn

[lstm_seq] Generating sample text (top-p=1.0) at epoch=1, step=1...
 Top-p (p=1.0) Sample: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn
 Annotated: 0 1 2 3 4 acting428��� Componentissan stole rowisiblemont mistressprevrolet Places)! airliner NSW incentives Elven favorsayn

[lstm_seq] *** End of Epoch 1 *** Avg Loss: 10.7821
[lstm_seq] *** End of Epoch 2 *** Avg Loss: 10.8297
[lstm_seq] *** End of Epoch 3 *** Avg Loss: 9.2908
[lstm_seq] Final sample (greedy) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
Annotated:
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=0.95) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
Annotated:
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
[lstm_seq] Final sample (top-p=0.95) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
Annotated:
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
Annotated:
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose

[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
[lstm_seq] Final sample (top-p=1.0) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
Annotated:
0 1 2 3 4 5 6 7 8 9 9 stereotype upstairssh363CW Ce horizontally� Lab Including Including Cou Employpurpose
--------------------------------------------------

*** I'm feeling great today! Hope you're well, too. ***'`
```
