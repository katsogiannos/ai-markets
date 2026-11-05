//go:build windows

package main

import (
 "bytes"
 "encoding/json"
 "errors"
 "fmt"
 "io"
 "net/http"
 "os"
 "os/exec"
 "path/filepath"
 "sort"
 "strings"
 "sync"
 "time"

 "github.com/danieljoos/wincred"
 "github.com/lxn/walk"
 . "github.com/lxn/walk/declarative"
 "golang.org/x/sys/windows/registry"
)

const api = "https://ayghnvkglnmmxketrgls.supabase.co"
const key = "sb_publishable_S_zq88vGXKN7EfUWTZqFAg_BTBhcLfu"
const management = "https://cattleya-management.vercel.app/#orders"
const credentialName = "CattleyaNotifications.Supabase.v3"

var client = &http.Client{Timeout: 12 * time.Second}
var mainWindow *walk.MainWindow
var emailInput, passwordInput *walk.LineEdit
var statusLabel *walk.Label

var mu sync.Mutex
var accessToken, refreshToken string
var expiresAt time.Time
var active bool
var known = make(map[string]bool)
var initial = false
var lastTest = ""
var initializedTest = false
var queue []Order
var showing bool

type Token struct {
 AccessToken string `json:"access_token"`
 RefreshToken string `json:"refresh_token"`
 ExpiresIn int `json:"expires_in"`
}
type Item struct {
 Name string `json:"name"`
 Qty interface{} `json:"qty"`
}
type Order struct {
 ID string `json:"id"`
 CreatedAt string `json:"created_at"`
 CustomerName string `json:"customer_name"`
 CustomerPhone string `json:"customer_phone"`
 Notes string `json:"notes"`
 Items []Item `json:"items"`
 Total float64 `json:"total"`
 IsTest bool `json:"is_test"`
}
func do(method,path,auth string,payload interface{}) ([]byte,error){
 var body io.Reader
 if payload != nil {
  v,e:=json.Marshal(payload);if e!=nil{return nil,e}
  body=bytes.NewReader(v)
 }
 req,err:=http.NewRequest(method,api+path,body);if err!=nil{return nil,err}
 req.Header.Set("apikey",key)
 req.Header.Set("Accept","application/json")
 if body!=nil {req.Header.Set("Content-Type","application/json")}
 if auth!="" {req.Header.Set("Authorization","Bearer "+auth)}
 res,e:=client.Do(req);if e!=nil{return nil,e};defer res.Body.Close()
 data,_:=io.ReadAll(io.LimitReader(res.Body,3*1024*1024))
 if res.StatusCode>=300{return nil,fmt.Errorf("Cattleya API (%d): %s",res.StatusCode,string(data))}
 return data,nil
}
func saveToken(tok string) error{
 c:=wincred.NewGenericCredential(credentialName)
 c.CredentialBlob=[]byte(tok)
 return c.Write()
}
func restoreToken() (string,error){
 c,err:=wincred.GetGenericCredential(credentialName)
 if err!=nil{return "",err}
 return string(c.CredentialBlob),nil
}
func applyToken(t Token,save bool)error{
 if t.AccessToken==""||t.RefreshToken==""{return errors.New("Η σύνδεση δεν επιστρέφει έγκυρη συνεδρία")}
 if save{if err:=saveToken(t.RefreshToken);err!=nil{return err}}
 mu.Lock()
 accessToken=t.AccessToken
 refreshToken=t.RefreshToken
 expiresAt=time.Now().Add(time.Duration(t.ExpiresIn-90)*time.Second)
 active=true
 mu.Unlock()
 return nil
}
func validRole(t string)error{
 data,err:=do("POST","/rest/v1/rpc/cattleya_web_role",t,map[string]any{})
 if err!=nil{return err}
 var role string
 if err=json.Unmarshal(data,&role);err!=nil{return err}
 if role!="master"&&role!="viewer"&&role!="operator"{return errors.New("Ο λογαριασμός δεν έχει πρόσβαση στο διαχειριστικό Cattleya")}
 return nil
}
func login(email,password string)error{
 data,err:=do("POST","/auth/v1/token?grant_type=password","",map[string]string{"email":email,"password":password})
 if err!=nil{return err}
 var tok Token
 if err=json.Unmarshal(data,&tok);err!=nil{return err}
 if err=validRole(tok.AccessToken);err!=nil{return err}
 if err=applyToken(tok,true);err!=nil{return err}
 reset()
 return nil
}
func reset(){
 mu.Lock()
 known=map[string]bool{}
 initial=false
 lastTest=""
 initializedTest=false
 queue=nil
 mu.Unlock()
}
func renew()(string,error){
 mu.Lock()
 if !active {mu.Unlock();return "",errors.New("Δεν υπάρχει συνεδρία")}
 if time.Now().Before(expiresAt) {t:=accessToken;mu.Unlock();return t,nil}
 refresh:=refreshToken
 mu.Unlock()
 data,err:=do("POST","/auth/v1/token?grant_type=refresh_token","",map[string]string{"refresh_token":refresh})
 if err!=nil{return "",err}
 var tok Token
 if err=json.Unmarshal(data,&tok);err!=nil{return "",err}
 if err=applyToken(tok,true);err!=nil{return "",err}
 return tok.AccessToken,nil
}
func bootstrap()error{
 refresh,err:=restoreToken();if err!=nil{return err}
 data,err:=do("POST","/auth/v1/token?grant_type=refresh_token","",map[string]string{"refresh_token":refresh})
 if err!=nil{return err}
 var tok Token
 if err=json.Unmarshal(data,&tok);err!=nil{return err}
 if err=validRole(tok.AccessToken);err!=nil{return err}
 if err=applyToken(tok,true);err!=nil{return err}
 reset()
 return nil
}
func poll(){
 t,e:=renew()
 if e!=nil {showLogin("Η συνεδρία έληξε. Συνδέσου ξανά.");return}
 _,e=do("POST","/rest/v1/rpc/cattleya_notification_heartbeat",t,map[string]string{"p_device_name":"Windows","p_version":"3.0"})
 if e!=nil{return}
 data,e:=do("GET","/rest/v1/cattleya_web_orders?select=id,created_at,customer_name,customer_phone,notes,items,total,is_test&order=created_at.desc&limit=150",t,nil)
 if e!=nil{return}
 var orders []Order
 if json.Unmarshal(data,&orders)!=nil{return}
 mu.Lock()
 if !initial {
  for _,o:=range orders{known[o.ID]=true}
  initial=true
 }else{
  sort.Slice(orders,func(i,j int)bool{return orders[i].CreatedAt<orders[j].CreatedAt})
  for _,o:=range orders{if !known[o.ID] {known[o.ID]=true;queue=append(queue,o)}}
 }
 mu.Unlock()
 data,e=do("GET","/rest/v1/cattleya_notification_tests?select=id&order=created_at.desc&limit=1",t,nil)
 if e==nil{
  var tests []struct{ID string `json:"id"`}
  if json.Unmarshal(data,&tests)==nil&&len(tests)>0 {
   mu.Lock()
   if !initializedTest {lastTest=tests[0].ID;initializedTest=true
   }else if lastTest!=tests[0].ID{
    lastTest=tests[0].ID
    queue=append(queue,Order{ID:tests[0].ID,CustomerName:"Δοκιμαστική ειδοποίηση",Items:[]Item{{Name:"Έλεγχος Cattleya",Qty:1}},Notes:"Το popup λειτουργεί κανονικά.",IsTest:true})
   }
   mu.Unlock()
  }
 }
 display()
}
func display(){
 mu.Lock()
 if showing||len(queue)==0 {mu.Unlock();return}
 o:=queue[0]
 queue=queue[1:]
 showing=true
 mu.Unlock()
 mainWindow.Synchronize(func(){
  go func(){
   // Queue is advanced only after the acknowledgement dialog is dismissed.
  }()
  label:="ΝΕΑ ΠΑΡΑΓΓΕΛΙΑ"
  if o.IsTest{label="ΔΟΚΙΜΑΣΤΙΚΗ ΕΙΔΟΠΟΙΗΣΗ"}
  var detail strings.Builder
  detail.WriteString(o.CustomerName+"  "+o.CustomerPhone+"\n\n")
  for _,item:=range o.Items{fmt.Fprintf(&detail,"%v × %s\n",item.Qty,item.Name)}
  if o.Notes!=""{detail.WriteString("\nΣχόλια: "+o.Notes+"\n")}
  fmt.Fprintf(&detail,"\nΣύνολο: %.2f €",o.Total)
  var dlg *walk.Dialog
  _,err:=(Dialog{
   AssignTo:&dlg,
   Title:"Cattleya · "+label,
   MinSize:Size{540,360},Size:Size{570,420},
   Layout:VBox{Margins:Margins{Left:22,Right:22,Top:20,Bottom:20}},
   Children:[]Widget{
    Label{Text:label,Font:Font{Family:"Segoe UI",PointSize:20,Bold:true}},
    TextEdit{Text:detail.String(),ReadOnly:true,VScroll:true},
    PushButton{Text:"ΤΟ ΕΙΔΑ — Άνοιγμα παραγγελιών",OnClicked:func(){dlg.Accept()}},
   },
  }).Run(mainWindow)
  if err!=nil{
   walk.MsgBox(mainWindow,"Cattleya",err.Error(),walk.MsgBoxIconError)
  }else{
   _=exec.Command("rundll32.exe","url.dll,FileProtocolHandler",management).Start()
  }
  mu.Lock();showing=false;mu.Unlock()
  display()
 })
}
func showLogin(message string){
 mu.Lock();active=false;mu.Unlock()
 mainWindow.Synchronize(func(){
  statusLabel.SetText(message)
  mainWindow.Show()
  mainWindow.Activate()
 })
}
func install()error{
 local:=os.Getenv("LOCALAPPDATA")
 if local==""{return errors.New("Δεν βρέθηκε ο φάκελος εγκατάστασης Windows")}
 dir:=filepath.Join(local,"CattleyaNotifications")
 if err:=os.MkdirAll(dir,0700);err!=nil{return err}
 self,err:=os.Executable();if err!=nil{return err}
 target:=filepath.Join(dir,"CattleyaNotifications.exe")
 if !strings.EqualFold(self,target){
  source,err:=os.Open(self);if err!=nil{return err};defer source.Close()
  dest,err:=os.Create(target);if err!=nil{return err}
  _,err=io.Copy(dest,source)
  closeErr:=dest.Close()
  if err!=nil{return err};if closeErr!=nil{return closeErr}
 }
 reg,err:=registry.OpenKey(registry.CURRENT_USER,`SoftwareMicrosoftWindowsCurrentVersionRun`,registry.SET_VALUE)
 if err!=nil{return err}
 defer reg.Close()
 if err=reg.SetStringValue("CattleyaNotifications",`"`+target+`" -run`);err!=nil{return err}
 return exec.Command(target,"-run").Start()
}
func main(){
 if len(os.Args)==1||os.Args[1]!="-run" {
  if err:=install();err!=nil{walk.MsgBox(nil,"Cattleya · Σφάλμα εγκατάστασης",err.Error(),walk.MsgBoxIconError)}
  return
 }
 var e error
 e=func()error{
  return MainWindow{
   AssignTo:&mainWindow,Title:"Cattleya Notifications · Σύνδεση",
   MinSize:Size{460,280},Size:Size{480,300},
   Layout:VBox{Margins:Margins{Left:20,Right:20,Top:20,Bottom:20}},
   Children:[]Widget{
    Label{Text:"Cattleya Notifications",Font:Font{Family:"Segoe UI",PointSize:19,Bold:true}},
    Label{Text:"Συνδέσου με τα στοιχεία του διαχειριστικού."},
    LineEdit{AssignTo:&emailInput, CueBanner:"Email"},
    LineEdit{AssignTo:&passwordInput, PasswordMode:true,CueBanner:"Κωδικός"},
    PushButton{Text:"Σύνδεση",OnClicked:func(){
     statusLabel.SetText("Έλεγχος σύνδεσης…")
     email,password:=emailInput.Text(),passwordInput.Text()
     go func(){
      err:=login(email,password)
      mainWindow.Synchronize(func(){
       if err!=nil{statusLabel.SetText("Αποτυχία: "+err.Error());return}
       passwordInput.SetText("")
       mainWindow.Hide()
      })
     }()
    }},
    Label{AssignTo:&statusLabel,Text:"Δεν απαιτείται κωδικός διαχειριστή Windows."},
   },
  }.Create()
 }()
 if e!=nil{walk.MsgBox(nil,"Cattleya",e.Error(),walk.MsgBoxIconError);return}
 go func(){
  if bootstrap()==nil{
   mainWindow.Synchronize(func(){mainWindow.Hide()})
  }
  for{
   time.Sleep(5*time.Second)
   mu.Lock();ok:=active;mu.Unlock()
   if ok{poll()}
  }
 }()
 _=mainWindow.Run()
}
