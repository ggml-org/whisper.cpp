# Language Server

This example consists of a simple language server to expose both unguided
and guided (command) transcriptions by sending json messages over stdout/stdin
as well as a rather robust vim plugin that makes use of the language server.

## Vim plugin quick start

Compile the language server with

```bash
make lsp
```
Install the plugin itself by copying or symlinking whisper.vim into ~/.vim/autoload/

In your vimrc, set the path of your whisper.cpp directory and optionally add some keybinds.

```vim
let g:whisper_dir = "~/whisper.cpp"
" Start listening for commands when Ctrl - g is pressed in normal mode
nnoremap <C-G> call whisper#requestCommands()<CR>
" Start unguided transcription when Ctrl - g is pressed in insert mode
inoremap <C-G> <Cmd>call whisper#doTranscription()<CR>
```

## Vim plugin usage

The vim plugin was designed to closely follow the mnemonics of vim

`s:spoken_dict` is used to translate keys to their spoken form.


Keys corresponding to a string use that spoken value normally and when a motion is expected, but use the key itself when a character is expected.  
Keys corresponding to a dict, like `i`, can have manual difinitions given to each possible commandset.

0 is normal (insert), 1 is motion (inside), 2 is it's usage as a single key ([till] i), and 3 is it's usage in an area selection (s -> [around] sentence)

Some punctuation items, like `-` are explicitly given pronunciations to prevent them from being picked as punctuation instead of an actual command word.

Not all commands will tokenize to a single token and this can interfere with interpretation. "yank" as an example, takes multiple tokens and correspondingly, will give more accurate detection when only the first "ya" is used. While it could be changed to something else that is a single token (copy), value was placed on maintaining vim mnemonics.

Commands that would normally move the editor into insert mode (insert, append, open, change) will begin unguided transcription.
Unguided transcription will end when a speech segment ends in exit.
Presence of punctuation can be designated by whether or not you add a pause between the previous speech segment and exit.
Exiting only occurs if exit is the last word, so "Take the first exit on your right" would not cause transcription to end.

After a command is evaluated, the plugin will continue listening for the next command.

While in command mode, "Exit" will end listening.

A best effort approach is taken to keep track of audio that is recorded while a previous chunk is still processing and immediately interpret it afterwards, but the current voice detection still needs a fairly sizable gap to determine when a command has been spoken.

Log information is sent to a special `whisper_log` buffer and can be accessed with
```vim
:e whisper_log
```

## Vim plugin configuration

`g:whisper_dir`  
A full path to the whisper.cpp repo. It can be expanded in the definition like so:
```vim
let g:whisper_dir = expand("~/whisper.cpp/")
```
(The WHISPER_CPP_HOME environment variable is also checked for users of the existing whisper.nvim script)

`g:whisper_lsp_path`  
Can be used to manually set the path to the language server.
If not defined, it will be inferred from the above whisper_dir

`g:whisper_model_path`  
A full path to the model to load. If not defined, it will default to ggml-base.en.bin

`g:whisper_user_commands`  
A dictionary of spoken commands that correspond to either strings or funcrefs.
This can be used to create connections with other user plugins, for example
```vim
let g:whisper_user_commands = {"gen": "llama#doLlamaGen"}
```
will trigger the llama.cpp plugin to begin generation when "gen" is spoken

## Language server methods

`registerCommandset`  
`params` is a list of strings that should be checked for with this commandset. The server prepends a space to these strings before tokenizing.  
Responds with  
`result.index` an integer index for the commandset registered, which should be included when initiating a guided transcription to select this commandset.
Will return an error if any of the commands in the commandset have duplicate tokenizations

`guided`  
`params.commandset_index` An index returned by a corresponding commandset registration. If not set, the most recently registered commandset is used.
`params.timestamp` A positive unsigned integer which designates a point in time which audio should begin processing from. If left blank, the start point of audio processing will be the moment the message is recieved. This should be left blank unless you have a timestamp from a previous response.  
Responds with  
`result.command_index` The numerical index (starting from 0) of the detected command in the selected commandset
`result.command_text` A string containing the command as provided in the commandset
`result.timestamp` A positive unsigned integer that designates the point in time which audio stopped being processed at. Pass this timestamp back in a subsequent message to mask the latency of transcription.

`unguided`  
`params.no_context` Sets the corresponding whisper `no_context` param. Defaults to true. Might provide more accurate results for consecutive unguided transcriptions if those after the first are set to false.
`params.prompt` If provided, sets the initial prompt used during transcription.
`params.timestamp` A positive unsigned integer which designates a point in time which audio should begin processing from. If left blank, the start point of audio processing will be the moment the message is recieved. This should be left blank unless you have a timestamp from a previous response.  
Responds with  
`result.transcription` A string containing the transcribed text.  N.B. This will almost always start with a space due to how text is tokenized.
`result.timestamp` A positive unsigned integer that designates the point in time which audio stopped being processed at. Pass this timestamp back in a subsequent message to mask the latency of transcription.
