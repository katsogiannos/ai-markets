$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ErrorActionPreference = 'Stop'
$api = 'https://ayghnvkglnmmxketrgls.supabase.co'
$key = 'sb_publishable_S_zq88vGXKN7EfUWTZqFAg_BTBhcLfu'
$manage = 'https://cattleya-management.vercel.app/#orders'
$homeFolder = Join-Path $env:APPDATA 'CattleyaNotifications'
New-Item -ItemType Directory -Force -Path $homeFolder | Out-Null
$sessionFile = Join-Path $homeFolder 'session.dpapi'
$script:token = ''
$script:refreshToken = ''
$script:expiry = [datetime]::MinValue
$script:online = $false
$script:known = @{}
$script:bootstrapped = $false
$script:lastTest = ''
$script:queue = New-Object System.Collections.Queue
$script:current = $null
$script:busy = $false

function Call-Post($path, $payload, $signed=$true) {
 $headers=@{apikey=$key;Accept='application/json'}
 if($signed){$headers.Authorization='Bearer '+$script:token}
 $bytes=[Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject $payload -Compress -Depth 12))
 return Invoke-RestMethod -Uri ($api+$path) -Headers $headers -Method POST -ContentType 'application/json' -Body $bytes -TimeoutSec 15
}
function Call-Get($path) {
 return Invoke-RestMethod -Uri ($api+$path) -Headers @{apikey=$key;Authorization=('Bearer '+$script:token);Accept='application/json'} -TimeoutSec 15
}
function Save-Session {
 if($script:refreshToken) {
  $safe=ConvertTo-SecureString -String $script:refreshToken -AsPlainText -Force
  ConvertFrom-SecureString -SecureString $safe | Set-Content -Path $sessionFile -Encoding UTF8
 }
}
function Refresh-Session {
 if([datetime]::UtcNow -lt $script:expiry){return}
 if(!$script:refreshToken){throw 'Χρειάζεται σύνδεση λογαριασμού.'}
 $r=Call-Post '/auth/v1/token?grant_type=refresh_token' @{refresh_token=$script:refreshToken} $false
 $script:token=[string]$r.access_token
 $script:refreshToken=[string]$r.refresh_token
 $script:expiry=[datetime]::UtcNow.AddSeconds([Math]::Max(30,[int]$r.expires_in-120))
 Save-Session
}
function Validate-Role {
 $role=Call-Post '/rest/v1/rpc/cattleya_web_role' @{}
 if([string]$role -notin @('master','viewer','operator')){throw 'Ο λογαριασμός δεν έχει δικαίωμα πρόσβασης.'}
}

$tray=New-Object System.Windows.Forms.NotifyIcon
$tray.Icon=[Drawing.SystemIcons]::Information
$tray.Text='Cattleya Notifications'
$tray.Visible=$true
$menu=New-Object System.Windows.Forms.ContextMenuStrip
$menuOpen=$menu.Items.Add('Άνοιγμα παραγγελιών')
$menuOpen.Add_Click({Start-Process $manage})
$menuLogin=$menu.Items.Add('Σύνδεση / Αλλαγή χρήστη')
$menuExit=$menu.Items.Add('Έξοδος')
$tray.ContextMenuStrip=$menu
$tray.Add_DoubleClick({Start-Process $manage})

$login=New-Object System.Windows.Forms.Form
$login.Text='Cattleya Notifications | Σύνδεση'
$login.Size=New-Object Drawing.Size(470,320)
$login.StartPosition='CenterScreen'
$login.FormBorderStyle='FixedDialog'
$login.MaximizeBox=$false
$login.BackColor=[Drawing.Color]::FromArgb(248,241,236)
$heading=New-Object System.Windows.Forms.Label
$heading.Text='Cattleya Notifications'
$heading.Font=New-Object Drawing.Font('Georgia',22)
$heading.SetBounds(25,18,400,46)
$login.Controls.Add($heading)
$labelEmail=New-Object Windows.Forms.Label
$labelEmail.Text='Email'
$labelEmail.SetBounds(25,75,100,20);$login.Controls.Add($labelEmail)
$email=New-Object Windows.Forms.TextBox
$email.SetBounds(25,97,405,28);$login.Controls.Add($email)
$labelPass=New-Object Windows.Forms.Label
$labelPass.Text='Κωδικός'
$labelPass.SetBounds(25,134,100,20);$login.Controls.Add($labelPass)
$pass=New-Object Windows.Forms.TextBox
$pass.UseSystemPasswordChar=$true
$pass.SetBounds(25,155,405,28);$login.Controls.Add($pass)
$submit=New-Object Windows.Forms.Button
$submit.Text='Σύνδεση'
$submit.Font=New-Object Drawing.Font('Segoe UI',10,[Drawing.FontStyle]::Bold)
$submit.BackColor=[Drawing.Color]::FromArgb(207,227,216)
$submit.SetBounds(25,195,405,40);$login.Controls.Add($submit)
$loginStatus=New-Object Windows.Forms.Label
$loginStatus.Text='Χρησιμοποίησε τα στοιχεία του διαχειριστικού.'
$loginStatus.SetBounds(25,245,405,40);$login.Controls.Add($loginStatus)

$popup=New-Object System.Windows.Forms.Form
$popup.Text='Cattleya | Νέα παραγγελία'
$popup.Size=New-Object Drawing.Size(570,535)
$popup.StartPosition='CenterScreen'
$popup.FormBorderStyle='FixedDialog'
$popup.MaximizeBox=$false
$popup.TopMost=$true
$popup.ShowInTaskbar=$true
$popup.BackColor=[Drawing.Color]::FromArgb(255,249,245)
$popTitle=New-Object Windows.Forms.Label
$popTitle.Text='ΝΕΑ ΠΑΡΑΓΓΕΛΙΑ'
$popTitle.Font=New-Object Drawing.Font('Georgia',22,[Drawing.FontStyle]::Bold)
$popTitle.SetBounds(22,18,515,50);$popup.Controls.Add($popTitle)
$popName=New-Object Windows.Forms.Label
$popName.Font=New-Object Drawing.Font('Segoe UI',12)
$popName.SetBounds(22,83,515,52);$popup.Controls.Add($popName)
$popItems=New-Object Windows.Forms.TextBox
$popItems.Multiline=$true;$popItems.ReadOnly=$true
$popItems.ScrollBars='Vertical'
$popItems.Font=New-Object Drawing.Font('Segoe UI',12)
$popItems.BackColor=[Drawing.Color]::FromArgb(207,227,216)
$popItems.SetBounds(22,144,515,205);$popup.Controls.Add($popItems)
$popTotal=New-Object Windows.Forms.Label
$popTotal.Font=New-Object Drawing.Font('Georgia',19)
$popTotal.SetBounds(22,362,515,44);$popup.Controls.Add($popTotal)
$ack=New-Object Windows.Forms.Button
$ack.Text='ΤΟ ΕΙΔΑ — Άνοιγμα παραγγελιών'
$ack.BackColor=[Drawing.Color]::FromArgb(207,227,216)
$ack.Font=New-Object Drawing.Font('Segoe UI',11,[Drawing.FontStyle]::Bold)
$ack.SetBounds(22,424,515,48);$popup.Controls.Add($ack)

function Display-Next {
 if($script:current -or $script:queue.Count -eq 0){return}
 $script:current=$script:queue.Dequeue()
 $o=$script:current
 $popTitle.Text=if($o.is_test){'ΔΟΚΙΜΑΣΤΙΚΗ ΕΙΔΟΠΟΙΗΣΗ'}else{'ΝΕΑ ΠΑΡΑΓΓΕΛΙΑ'}
 $popName.Text=[string]$o.customer_name + ' | ' + [string]$o.customer_phone
 $lines=New-Object 'System.Collections.Generic.List[string]'
 foreach($i in @($o.items)) {
  if($null -ne $i){$lines.Add([string]$i.qty+' × '+[string]$i.name)}
 }
 if($o.notes){$lines.Add('Σχόλια: '+[string]$o.notes)}
 $popItems.Text=($lines -join [Environment]::NewLine)
 $popTotal.Text='Σύνολο: '+[string]$o.total+' €'
 $popup.Show()
 $popup.TopMost=$true
 $popup.Activate()
 [System.Media.SystemSounds]::Exclamation.Play()
}
function Acknowledge {
 $popup.Hide()
 $script:current=$null
 Start-Process $manage
 Display-Next
}
$ack.Add_Click({Acknowledge})
$popup.Add_FormClosing({$_.Cancel=$true;Acknowledge})
$menuLogin.Add_Click({$login.Show();$login.Activate()})
$login.Add_FormClosing({$_.Cancel=$true;$login.Hide()})
$timer=New-Object Windows.Forms.Timer
$timer.Interval=5000
$alarm=New-Object Windows.Forms.Timer
$alarm.Interval=5000
$alarm.Add_Tick({if($popup.Visible){[System.Media.SystemSounds]::Exclamation.Play()}})
$alarm.Start()
$menuExit.Add_Click({
 $timer.Stop();$alarm.Stop();$tray.Visible=$false
 [Windows.Forms.Application]::Exit()
})

function Poll {
 if(!$script:online -or $script:busy){return}
 $script:busy=$true
 try{
  Refresh-Session
  Call-Post '/rest/v1/rpc/cattleya_notification_heartbeat' @{p_device_name=$env:COMPUTERNAME;p_version='2.0'} | Out-Null
  $rows=@(Call-Get '/rest/v1/cattleya_web_orders?select=id,created_at,customer_name,customer_phone,notes,items,total,is_test&order=created_at.desc&limit=150')
  if(!$script:bootstrapped){
   foreach($o in $rows){$script:known[[string]$o.id]=$true}
   $script:bootstrapped=$true
  }else{
   $new=@($rows|Where-Object {-not $script:known.ContainsKey([string]$_.id)})
   [array]::Reverse($new)
   foreach($o in $new) {
    $script:known[[string]$o.id]=$true
    $script:queue.Enqueue($o)
   }
  }
  $tests=@(Call-Get '/rest/v1/cattleya_notification_tests?select=id,created_at&order=created_at.desc&limit=1')
  if($tests.Count -and $tests[0].id){
   $id=[string]$tests[0].id
   if(!$script:lastTest){$script:lastTest=$id}
   elseif($script:lastTest -ne $id){
    $script:lastTest=$id
    $script:queue.Enqueue([pscustomobject]@{
     customer_name='Δοκιμή ειδοποιήσεων'
     customer_phone=''
     notes='Το popup λειτουργεί κανονικά.'
     items=@([pscustomobject]@{qty=1;name='Δοκιμαστικό προϊόν'})
     total='0,00'
     is_test=$true
    })
   }
  }
  Display-Next
  $tray.Text='Cattleya — Συνδεδεμένο'
 } catch {
  $tray.Text='Cattleya — Πρόβλημα σύνδεσης'
 } finally {$script:busy=$false}
}
$timer.Add_Tick({Poll})
$submit.Add_Click({
 try{
  $submit.Enabled=$false;$loginStatus.Text='Σύνδεση…'
  $r=Call-Post '/auth/v1/token?grant_type=password' @{email=$email.Text.Trim();password=$pass.Text} $false
  $script:token=[string]$r.access_token
  $script:refreshToken=[string]$r.refresh_token
  $script:expiry=[datetime]::UtcNow.AddSeconds([Math]::Max(30,[int]$r.expires_in-120))
  Validate-Role
  Save-Session
  $script:known=@{};$script:bootstrapped=$false;$script:lastTest=''
  $script:online=$true
  $pass.Clear();$login.Hide()
  $timer.Start();Poll
  $tray.ShowBalloonTip(1600,'Cattleya','Οι ειδοποιήσεις είναι ενεργές.','Info')
 }catch{
  $loginStatus.Text='Αποτυχία: '+$_.Exception.Message
 }finally{$submit.Enabled=$true}
})
if(Test-Path $sessionFile){
 try{
  $safe=Get-Content -Path $sessionFile | ConvertTo-SecureString
  $bstr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($safe)
  try{$script:refreshToken=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)}
  finally{[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)}
  Refresh-Session
  Validate-Role
  $script:online=$true
  $timer.Start()
  Poll
 }catch{$script:online=$false}
}
if(!$script:online){$login.Show()}
[System.Windows.Forms.Application]::Run((New-Object Windows.Forms.ApplicationContext))
