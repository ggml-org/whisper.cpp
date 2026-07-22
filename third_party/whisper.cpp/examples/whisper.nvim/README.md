# whisper.nvim

Speech-to-text in Neovim

The transcription is performed on the CPU and no data leaves your computer. Works best on Apple Silicon devices.

https://user-images.githubusercontent.com/1991296/198382564-784e9663-2037-4d04-99b8-f39136929b7e.mp4

## Usage

- Simply press `Ctrl-G` in `INSERT`, `VISUAL` or `NORMAL` mode and say something
- When you are done - press `Ctrl-C` to end the transcription and insert the transcribed text under the cursor

## Installation

*Note: this is a bit tedious and hacky atm, but I hope it will be improved with time*

- Clone this repo and build the `stream` tool:

  ```
  git clone https://github.com/ggerganov/whisper.cpp
  cd whisper.cpp
  make stream
  ```

- Download the `base.en` Whisper model (140 MB):

  ```
  ./models/download-ggml-model.sh base.en
  ```

- Place the [whisper.nvim](whisper.nvim) script somewhere in your PATH and give it execute permissions:

  ```
  cp examples/whisper.nvim/whisper.nvim ~/bin/
  chmod u+x ~/bin/whisper.nvim
  ```

- Fine-tune the script to your preference and machine parameters:

  ```
  ./stream -t 8 -m models/ggml-base.en.bin --step 350 --length 10000 -f /tmp/whisper.nvim 2> /dev/null
  ```

  On slower machines, try to increase the `step` parameter.

- Add the following shortcuts to your `~/.config/nvim/init.vim`:

  ```
  inoremap <C-G>  <C-O>:!whisper.nvim<CR><C-O>:let @a = system("cat /tmp/whisper.nvim \| tail -n 1 \| xargs -0 \| tr -d '\\n' \| sed -e 's/^[[:space:]]*//'")<CR><C-R>a
  nnoremap <C-G>       :!whisper.nvim<CR>:let @a = system("cat /tmp/whisper.nvim \| tail -n 1 \| xargs -0 \| tr -d '\\n' \| sed -e 's/^[[:space:]]*//'")<CR>"ap
  vnoremap <C-G> c<C-O>:!whisper.nvim<CR><C-O>:let @a = system("cat /tmp/whisper.nvim \| tail -n 1 \| xargs -0 \| tr -d '\\n' \| sed -e 's/^[[:space:]]*//'")<CR><C-R>a
  ```
  
  Explanation: pressing `Ctrl-G` runs the [whisper.nvim](whisper.nvim) script which in turn calls the `stream` binary to transcribe your speech through the microphone. The results from the transcription are continuously dumped into `/tmp/whisper.nvim`. After you kill the program with `Ctrl-C`, the vim command grabs the last line from the `/tmp/whisper.nvim` file and puts it under the cursor.
  
  Probably there is a much more intelligent way to achieve all this, but this is what I could hack in an hour. Any suggestions how to improve this are welcome.
  
You are now ready to use speech-to-text in Neovim!

## TODO

There are a lot of ways to improve this idea and I don't have much experience with Vim plugin programming, so contributions are welcome! 

- [ ] **Wrap this into a plugin**
  
  It would be great to make a standalone plugin out of this that can be installed with `vim-plug` or similar
  
- [ ] **Simplify the `init.vim` mappings (maybe factor out the common call into a separate function)**
- [ ] **Add Copilot/GPT-3 integration**

  This is probably a very long shot, but I think it will be very cool to have the functionality to select some code and then hit Ctrl-G and say something like:
  
  *"refactor this using stl containers"*
  
  or
  
  *"optimize by sorting the data first"*
  
  The plugin would then make an appropriate query using the selected text and code context to Copilot or GPT-3 and return the result.
  
  Here is a proof-of-concept:
  
  https://user-images.githubusercontent.com/1991296/199078847-0278fcde-5667-4748-ba0d-7d55381d6047.mp4
    
  https://user-images.githubusercontent.com/1991296/200067939-f98d2ac2-7519-438a-85f9-79db0841ba4f.mp4
  
  For explanation how this works see: https://twitter.com/ggerganov/status/1587168771789258756

## Discussion

If you find this idea interesting, you can join the discussion here: https://github.com/ggerganov/whisper.cpp/discussions/108
