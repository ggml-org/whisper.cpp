if !exists("g:whisper_dir")
    let g:whisper_dir = expand($WHISPER_CPP_HOME)
    if g:whisper_dir == ""
        echoerr "Please provide a path to the whisper.cpp repo in either the $WHISPER_CPP_HOME environment variable, or g:whisper_dir"
    endif
endif
if !exists("g:whisper_lsp_path")
    let g:whisper_lsp_path = g:whisper_dir .. "lsp"
    if !filereadable(g:whisper_lsp_path)
        echoerr "Was not able to locate a lsp executable at: " .. g:whisper_lsp_path
        throw "Executable not found"
    endif
endif
if !exists("g:whisper_model_path")
    " TODO: allow custom paths relative to the repo dir
    let g:whisper_model_path = g:whisper_dir .. "models/ggml-base.en.bin"
    if !filereadable(g:whisper_model_path)
        echoerr "Could not find model at: " .. g:whisper_model_path
        throw "Model not found"
    endif
endif
let s:output_buffer = bufnr("whisper_log", v:true)
call setbufvar(s:output_buffer,"&buftype","nofile")
let s:lsp_command = [g:whisper_lsp_path,"-m",g:whisper_model_path]
" For faster execution. TODO: server load multiple models/run multiple servers?
" let s:lsp_command = [g:whisper_lsp_path, "-m", g:whisper_dir .. "models/ggml-tiny.en.bin", "-ac", "128"]

" requestCommands([params_dict])
func whisper#requestCommands(...)
    let l:req = {"method": "guided", "params": {"commandset_index": 0}}
    if a:0 > 0
        call extend(l:req.params, a:1)
    endif
    let resp = ch_sendexpr(g:lsp_job, l:req, {"callback": function("s:commandCallback", [l:req.params, 0])})
endfunction

" doTranscription([params_dict])
func whisper#doTranscription(...)
    let l:req = {"method": "unguided", "params": {}}
    if a:0 > 0
        call extend(l:req.params, a:1)
    endif
    let resp = ch_sendexpr(g:lsp_job, l:req, {"callback": function("s:transcriptionCallback", [function("s:insertText"),function("s:endTranscription")])})
endfunction

" For testing
func whisper#uppertest(cha)
    echo tr(a:cha, s:c_lowerkeys, s:c_upperkeys)
endfunction


" (upper, exit, count, motion, command, insert/append, save run) "base"
" (upper, exit, count, motion, command, inside/around)           "motion/visual"
" (upper, exit, count, motion, line,    inside/around)           "command already entered"
" (upper, exit, key,                                 )           "from/till"

" upper and lower keys is used to translate between cases with tr
" Must be sunchronized
let s:c_lowerkeys = "1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./\""
let s:c_upperkeys = "!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?'"
let s:c_count = split("1234567890\"",'\zs')
let s:c_command = split("ryuogpdxcv.iam", '\zs')
let s:c_motion = split("wetf'hjklnb$^)",'\zs')
" object words: Word, Sentence, Paragraph, [, (, <, Tag, {. ", '
let s:c_area = split("wsp])>t}\"'",'\zs')
"Special commands.
let s:c_special_always = ["exit", "upper"]
let s:c_special_normal = ["save", "run", "space"]

" If not in dict, key is spoken word,
" If key resolves to string, value is used for normal/motion, but key for chars
" If key resolves to dict, {0: "normal",1: "motion",2:"single char",3: "area"}
" Missing entries fall back as follows {0: "required", 1: 0, 2: "key", 3: 0}
let s:spoken_dict = {"w": "word", "e": "end", "r": "replace", "t": {0: "till", 3: "tag"}, "y": "yank", "u": "undo", "i": {0: "insert", 1: "inside"}, "o": "open", "p": {0: "paste", 3: "paragraph"},  "a": {0: "append", 1: "around"}, "s": {0: "substitute", 3: "sentence"}, "d": "delete", "f": "from", "g": "go", "h": "left", "j": "down", "k": "up", "l": "right", "c": "change", "v": "visual", "b": "back", "n": "next", "m": "mark", ".": {0: "repeat", 2: "period"}, "]": {0: "bracket", 2: "bracket"}, "'": {0: "jump", 2: "apostrophe", 3:  "apostrophe"}, '"': {0: 'register', 2: "quotation", 3: "quotation"}, "-": {0: "minus", 2: "minus"}, "$": {0: "dollar", 2: "dollar"}, "^": {0: "carrot", 2: "carrot"}, ")": {0: "sentence", 2: "parenthesis",  3: "parenthesis"}, "}": {0: "paragraph", 2: "brace", 3: "brace"}, ">": {0: "indent", 2: "angle", 3: "angle"}}

" Give this another pass. This seems overly hacky even if it's functional
let s:sub_tran_msg = ""
func s:subTranProg(msg)
    if s:sub_tran_msg != ""
        let s:sub_tran_msg = s:sub_tran_msg .. a:msg
        if mode() !=? 'v'
            exe "normal" "u" .. s:sub_tran_msg
        endif
    else
        if s:command_backlog == ""
            " this should not occur
            call s:logCallback(0, "Warning: Encountered sub transcription without prior command")
            let s:command_backlog = "a"
        endif
        if a:msg[0] == ' '
            let s:sub_tran_msg = s:command_backlog .. a:msg[1:-1]
        else
            let s:sub_tran_msg = s:command_backlog  .. a:msg
        endif
        if mode() !=? 'v'
            exe "normal" s:sub_tran_msg
        endif
    endif
    call appendbufline(s:output_buffer, "$", s:sub_tran_msg ..  ":" .. string(a:msg ))
endfunction

func s:subTranFinish(params, timestamp)
    let s:repeat_command = s:sub_tran_msg
    " Visual selection is lot if used with streaming, so streaming of partial
    " transcriptions is disabled in visual mode
    if mode() ==? 'v'
        exe "normal" s:sub_tran_msg
    endif
    let s:sub_tran_msg = ""
    let s:command_backlog = ""
    exe "normal a\<C-G>u"
    let l:params = a:params
    let l:params.timestamp = a:timestamp
    if exists("l:params.commandset_index")
        unlet l:params.commandset_index
    endif
    call whisper#requestCommands(a:params)
endfunction

func s:logCallback(channel, msg)
    call appendbufline(s:output_buffer,"$",a:msg)
endfunction


func s:transcriptionCallback(progressCallback, finishedCallback, channel, msg)
    let l:tr = a:msg.result.transcription

    let l:ex_ind = match(tolower(l:tr),"exit", len(l:tr)-6)
    " The worst case I've observed so far is " Exit.", which is 6 characters
    if l:ex_ind != -1
        call a:progressCallback(strpart(l:tr,0,l:ex_ind-1))
        call a:finishedCallback(a:msg.result.timestamp)
    else
        call a:progressCallback(l:tr)
        let req = {"method": "unguided", "params": {"timestamp": a:msg.result.timestamp, "no_context": v:true}}
        let resp = ch_sendexpr(g:lsp_job, req, {"callback": function("s:transcriptionCallback", [a:progressCallback, a:finishedCallback])})
    endif
endfunc
func s:insertText(msg)
    exe "normal a" .. a:msg
endfunction
func s:endTranscription(timestamp)
    call appendbufline(s:output_buffer, "$", "Ending unguided transcription")
endfunction



" If a command does not include a whole actionable step, attempting to execute
" it discards the remainder of things. There is likely a simpler solution,
" but it can be made functional now by storing a backbuffer until actionable
let s:command_backlog = ""
let s:repeat_command = ""
let s:preceeding_upper = v:false
func s:commandCallback(params, commandset_index, channel, msg)
    let l:command_index = a:msg.result.command_index
    let l:do_execute = v:false
    let l:next_mode = a:commandset_index
    let l:command = s:commandset_list[a:commandset_index][l:command_index]
    call s:logCallback(0, string(a:msg) .. " " .. a:commandset_index .. " " .. l:command)
    if l:command_index == 0
        "exit
        "if s:command_backlog == ""
        call s:logCallback(0,"Stopping command mode")
        echo "No longer listening"
        let s:command_backlog = ""
        return
        "else
        " Legacy code to clear an existing buffer with exit.
        " Was found to be rarely desired and is better introduced as a
        " standalone command (clear?)
        "   call s:logCallback(0,"Clearing command_backlog" .. s:command_backlog)
        "   let s:command_backlog = ""
        "   let s:preceeding_upper = v:false
        " endif
    elseif l:command_index == 1
        " upper
        let s:preceeding_upper = !s:preceeding_upper
    elseif l:command == "save"
        " save and run can only happen in commandset 0,
        exe "w"
    elseif l:command == "run"
        exe "make run"
    elseif l:command == "space"
        exe "normal i \<ESC>l"
    elseif has_key(s:c_user, l:command)
        let Userfunc = s:c_user[l:command]
        if type(Userfunc) == v:t_string
            let Userfunc = function(Userfunc)
        endif
        call Userfunc()
    else
        if s:preceeding_upper
            " Upper should keep commandset
            let s:preceeding_upper = v:false
            let l:visual_command = tr(l:command, s:c_lowerkeys, s:c_upperkeys)
        else
            let l:visual_command = l:command
        endif
        echo s:command_backlog .. " - " .. l:visual_command
        let s:command_backlog = s:command_backlog .. l:visual_command
        if a:commandset_index == 2 || a:commandset_index == 3
            " single key, either completes motion, replace, or register
            " Should move to execute unless part of a register
            " Change will be caught at execute
            if s:command_backlog[-2:-2] !=# '"'
                call s:logCallback(0,"not register")
                let l:do_execute = v:true
            end
            let l:next_mode = 0
            " commandset index only matters for a/i
        elseif (l:command == "a" || l:command == "i") && a:commandset_index == 1
            " inside/around. Is commandset 3
            let l:next_mode = 3
        elseif l:command ==# '"'
            let l:next_mode = 2
        elseif index(s:c_count, l:command) != -1
            let l:next_mode = a:commandset_index
        elseif index(s:c_motion, l:command) != -1
            if l:command == 't' || l:command == 'f' || l:command == "'"
                " prompt single key
                let l:next_mode = 2
            else
                let l:do_execute = v:true
                let l:next_mode = 0
            endif
        elseif index(s:c_command, l:command) != -1
            if index(["y","g","d","c"], s:command_backlog[-1:-1]) != -1 && s:command_backlog[-1:-1] != s:command_backlog[-2:-2] && mode() !=? 'v'
                " need motion or repeated command
                " Potential for bad state here if disparaging command keys are
                " entered (i.e. yd), but vim can handle checks for this at exe
                " And checking for cases like y123d would complicate things
                let l:next_mode = 1
            elseif index(["i","a","c", "o", "s"], l:command) != -1 || s:command_backlog[-1:-1] ==# 'R'
                "'Insert' mode, do general transcription
                let l:req = {"method": "unguided", "params": a:params}
                let l:req.params.timestamp = a:msg.result.timestamp
                let l:req.params.no_context = v:true
                let resp = ch_sendexpr(g:lsp_job, req, {"callback": function("s:transcriptionCallback", [function("s:subTranProg"), function("s:subTranFinish", [a:params])])})
                return
            elseif l:command == 'r' || l:command == 'm'
                let l:next_mode = 2
            elseif l:command == '.'
                let l:next_mode = 0
                let l:do_execute = v:true
                let s:command_backlog = s:command_backlog[0:-2] .. s:repeat_command
            else
                if l:command ==? 'v'
                    let l:next_mode = 1
                else
                    let l:next_mode = 0
                endif
                let l:do_execute = v:true
            endif
        else
            throw "Invalid command state: " .. l:command .. " " .. a:commandset_index .. " " .. s:command_backlog
        endif
    endif
    if l:do_execute
        if mode() ==?'v' && l:next_mode == 0
            let l:next_mode = 1
        elseif match(s:command_backlog, 'c') != -1
            let l:req = {"method": "unguided", "params": a:params}
            let l:req.params.timestamp = a:msg.result.timestamp
            let l:req.params.no_context = v:true
            let resp = ch_sendexpr(g:lsp_job, req, {"callback": function("s:transcriptionCallback", [function("s:subTranProg"), function("s:subTranFinish", [a:params])])})
            return
        endif
        exe "normal" s:command_backlog
        if index(s:c_motion + ["u"],l:command) == -1
            exe "normal a\<C-G>u"
            let s:repeat_command = s:command_backlog
            call s:logCallback(0, s:command_backlog)
        endif
        let s:command_backlog = ""
    endif
    let l:req = {"method": "guided", "params": a:params}
    let l:req.params.timestamp = a:msg.result.timestamp
    let l:req.params.commandset_index = l:next_mode
    let resp = ch_sendexpr(g:lsp_job, l:req, {"callback": function("s:commandCallback",[a:params, l:next_mode])})
endfunction

func s:loadedCallback(channel, msg)
    echo "Loading complete"
    call s:logCallback(a:channel, a:msg)
endfunction

func s:registerCommandset(commandlist, is_final)
    let req = {"method": "registerCommandset"}
    let req.params = a:commandlist
    call s:logCallback(0, join(a:commandlist))
    call add(g:whisper_commandlist_spoken, a:commandlist)
    if a:is_final
        let resp = ch_sendexpr(g:lsp_job, req, {"callback": "s:loadedCallback"})
    else
        let resp = ch_sendexpr(g:lsp_job, req, {"callback": "s:logCallback"})
    endif
endfunction

func s:registerAllCommands()
    let l:normal = s:c_special_always + s:c_special_normal + s:c_count + s:c_command + s:c_motion + keys(s:c_user)
    let l:visual = s:c_special_always + s:c_count + s:c_command + s:c_motion
    " Currently the same as visual.
    " let l:post_command = s:c_special_always + s:c_count + s:c_command + s:c_motion
    let l:single_key = s:c_special_always + split(s:c_lowerkeys, '\zs')
    let l:area = s:c_special_always + s:c_area

    " Used only for compatibility with the testing script
    let g:whisper_commandlist_spoken = []

    let s:commandset_list = [l:normal, l:visual, l:single_key, l:area]
    call s:registerCommandset(s:commandsetToSpoken(l:normal, 0), v:false)
    call s:registerCommandset(s:commandsetToSpoken(l:visual, 1), v:false)
    call s:registerCommandset(s:commandsetToSpoken(l:single_key, 2), v:false)
    call s:registerCommandset(s:commandsetToSpoken(l:area, 3), v:true)
endfunction

func s:commandsetToSpoken(commandset, spoken_index)
    let l:spoken_list = []
    for l:command in a:commandset
        if has_key(s:spoken_dict, l:command)
            let l:spoken_value = s:spoken_dict[l:command]
            if type(l:spoken_value) == v:t_dict
                if has_key(l:spoken_value, a:spoken_index)
                    let l:spoken_value = l:spoken_value[a:spoken_index]
                else
                    if a:spoken_index == 2
                        let l:spoken_value = l:command
                    else
                        let l:spoken_value = l:spoken_value[0]
                    endif
                endif
            else
                if a:spoken_index == 2
                    let l:spoken_value = l:command
                endif
            endif
        else
            let l:spoken_value = l:command
        endif
        call add(l:spoken_list, l:spoken_value)
    endfor
    return l:spoken_list
endfunction

" TODO: Check lifetime. If the script is resourced, is the existing
" s:lsp_job dropped and therefore killed?
" This seems to not be the case and I've had to deal with zombie processes
" that survive exiting vim, even though said behavior conflicts with my
" understanding of the provided documentation
let s:lsp_opts = {"in_mode": "lsp", "out_mode": "lsp", "err_mode": "nl", "err_io": "buffer", "err_buf": s:output_buffer}
if !exists("g:lsp_job")
    if exists("g:whisper_user_commands")
        let s:c_user = g:whisper_user_commands
    else
        let s:c_user = {}
    endif
    let g:lsp_job = job_start(s:lsp_command, s:lsp_opts)
    if job_status(g:lsp_job) == "fail"
        echoerr "Failed to start whisper job"
    endif
    call s:registerAllCommands()
endif
