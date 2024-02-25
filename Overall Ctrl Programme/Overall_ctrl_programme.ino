#include <Servo.h> 
#include <String.h>

// rgb init
int s0=4,s1=5,s2=6,s3=7;
int sensorOut=2;
int countR=0,countG=0,countB=0;
int flag=0;

void setup() {
    // Serial setup
    Serial.begin(9600);
    Serial1.begin(9600);
    
    // rgb setup
    pinMode(s0,OUTPUT);
    pinMode(s1,OUTPUT);
    pinMode(s2,OUTPUT);
    pinMode(s3,OUTPUT);
}


//fsr
int getPressure(int analogInput, String str) {
  int fsrReading = analogRead(analogInput); 
  delay(10);
  Serial.print(str); Serial.print(","); Serial.println(fsrReading);
  delay(10);
  return fsrReading;
}


//rgb
void read_rgb() {
  countR= pulseIn(sensorOut, LOW);
  Serial.print("red=");
  Serial.println(countR/10);
  delay(50);
  digitalWrite(s2,HIGH);
  digitalWrite(s3,HIGH);
  
  countG= pulseIn(sensorOut, LOW);
  Serial.print("green=");
  Serial.println(countG/10);
  delay(50);
  digitalWrite(s2,LOW);
  digitalWrite(s3,HIGH);
  
  countB= pulseIn(sensorOut, LOW);
  Serial.print("blue=");
  Serial.println(countB/10);
  delay(50);
  Serial.println("\n");
  digitalWrite(s2,LOW);
  digitalWrite(s3,LOW);
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
  Serial.println(number);
  return constrain(number, 0, 2500);
}

int input = 0;
String command="";

bool grasping=0;
int data =0;
void loop() {
  if (Serial.available()) {  // 如果串口0有数据可用
    data = Serial.parseInt(); // 读取串口0的数字输入
    if (data != 0) {
        switch(data){

        case 1:// initial phase
        {
          grasping=0;
          command=summon_command(1500,2000,200,600,-1);
          Serial1.print(command);
          break;
        }

        case 2:// reach out
        {
          grasping=0;
          command=summon_command(1500,1300,1400,600,-1);
          Serial1.print(command);
          break;
        }
          
        case 3:// grasp
        {
          command=summon_command(1500,1300,1400,600,1);
          Serial1.print(command);
          grasping=1;
          break;
        }

        case 4:// release
        {
          grasping=0;
          command=summon_command(1500,1300,1400,600,2499);
          Serial1.print(command);
          for(int i=0;i<5;i++){
            read_rgb();
          }
          break;
        }

        default:{
          grasping=0;
          Serial.print("wrong input\n");
          break;
        }
  }

    data=0;
    }
  }

if (grasping==1){
  getPressure(A0,"A0");
  getPressure(A1,"A1");
}
// MONKEY SAFE GUARD
  if (Serial1.available()) { 
    Serial.println("ERROR: Serial1 has output."); 
    // 清空串口1的发送缓冲区
    while (Serial1.available()) {
      Serial1.read(); // 读取串口1接收缓冲区中的数据，直到为空
    }
  
  }
}