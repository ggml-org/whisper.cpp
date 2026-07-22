# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
param(
  [Parameter(Mandatory=$true)][int]$voicenum,
  [Parameter(Mandatory=$true)][string]$textfile
)

Add-Type -AssemblyName System.Speech;
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$voiceoptions = $speak.GetInstalledVoices("en-US");
$voice = $voiceoptions[$voicenum % $voiceoptions.count];
$speak.SelectVoice($voice.VoiceInfo.Name);
$speak.Rate="0";
$text = Get-Content -Path $textfile;
$speak.Speak($text);
