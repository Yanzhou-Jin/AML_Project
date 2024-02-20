#include <Servo.h> 
#include <String.h>
Servo myservo;  // 创建Servo对象来控制舵机
void setup() {
  Serial.begin(9600);
}

// 可以测量10g-5公斤量程内的所有物体a
// 0N-50N

int MK_Port(int in){
  if (in>=49){
    return in;
  }
  return 0;
}

String get_command(String commandline, int ch, int pw, int spd, int time) {
  if (ch < 0) {
    return "";
  }
  
  commandline += "#" + String(ch);
  if (pw >= 0) {
    commandline += "P" + String(pw);
  }
  if (spd >= 0) {
    commandline += "S" + String(spd);
  }
  if (time >= 0) {
    commandline += "T" + String(time);
  }
  
  
  ch = -1;
  pw = -1;
  spd = -1;
  time = -1;
  
  return commandline;
}


String summon_command(int a, int b,int c,int d,int e){
  String commandline="";
  commandline=get_command(commandline,0,a,200,-1);
  commandline=get_command(commandline,1,b,200,-1);
  commandline=get_command(commandline,2,c,200,-1);
  commandline=get_command(commandline,3,d,200,-1);
  commandline=get_command(commandline,4,e,-1,-1);
  commandline += "<cr>";
  return commandline;
}
int readNumberFromSerial() {
  String numberString = Serial.readStringUntil('\n'); // 读取串口输入直到换行符为止
  
  int number = numberString.toInt();  // 将字符串转换为整数
  // 限制数字的范围在0到2500之间
  // Serial.print(number);
  return constrain(number, 0, 2500);
}

int flag=0;
char inByte=0;
int input=0;
String command="";
void loop() {

  command="";
  if (Serial.available() > 0) {
    // get incoming byte:
    // inByte = Serial.read();
    // flag=MK_Port(inByte);
    input=readNumberFromSerial();
  }
  if (input==2){//初始姿态，-1代表空，5个数代表5个关节的值，从下到上01234共5个关节，0-2500取值
    command=summon_command(1500,1300,1400,600,-1);
    Serial.println(command);
    input=0; 
  }
  if (input==1){//抓取姿态
    command=summon_command(1500,2000,200,600,-1);
    Serial.println(command);
    input=0; 
  }
  if (input==3){//抓
    command=summon_command(1500,1300,1400,600,1);
    Serial.println(command);
    input=0; 
  }
  if (input==4){//抓
    command=summon_command(1500,1300,1400,600,2499);
    Serial.println(command);
    input=0; 
  }
}

