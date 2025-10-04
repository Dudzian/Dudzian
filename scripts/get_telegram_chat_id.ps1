Param(
  [Parameter(Mandatory=$false)][string]$Token = $env:TELEGRAM_BOT_TOKEN
)
if (-not $Token) {
  Write-Error "Brak tokena. Użyj: .\scripts\get_telegram_chat_id.ps1 -Token <BOT_TOKEN> lub ustaw TELEGRAM_BOT_TOKEN w środowisku/.env"
  exit 1
}
$resp = Invoke-WebRequest -Uri ("https://api.telegram.org/bot{0}/getUpdates" -f $Token) -Method GET -TimeoutSec 15
$data = $resp.Content | ConvertFrom-Json
if (-not $data.ok) {
  Write-Error ("API error: {0}" -f ($data | ConvertTo-Json -Depth 5))
  exit 1
}
$ids = @()
foreach ($upd in $data.result) {
  foreach ($k in @("message","edited_message","channel_post","edited_channel_post","my_chat_member","chat_member")) {
    if ($upd.$k -ne $null) {
      $chat = $upd.$k.chat
      if ($chat -ne $null) {
        $ids += [PSCustomObject]@{
          id         = $chat.id
          type       = $chat.type
          title      = $chat.title
          username   = $chat.username
          first_name = $chat.first_name
          last_name  = $chat.last_name
        }
      }
    }
  }
}
$uniq = $ids | Sort-Object id -Unique
if ($uniq.Count -eq 0) {
  Write-Output "Brak chat_id w update'ach. Napisz wiadomość do bota / dodaj do grupy i spróbuj ponownie."
} else {
  $out = @{ timestamp_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"; found_chats = $uniq }
  $out | ConvertTo-Json -Depth 6
}
