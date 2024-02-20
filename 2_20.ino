#include <Servo.h> 
#include <String.h>
#include <FlexiTimer2.h>

// rgb init
int s0=4,s1=5,s2=6,s3=7;
int sensorOut=2;
byte counter=0;
int countR=0,countG=0,countB=0;
int flag=0;

// fsr init
Servo myservo; 
int angle;

void setup() {
    // motor setup
    Serial1.begin(9600);
    
    // fsr setup
    angle=180;
    myservo.write(angle);
    
    // rgb setup
    Serial.begin(9600);
    pinMode(s0,OUTPUT);
    pinMode(s1,OUTPUT);
    pinMode(s2,OUTPUT);
    pinMode(s3,OUTPUT);
}


//fsr
int getPressure(int analogInput, String str) {
  int fsrReading = analogRead(analogInput); 
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

// control
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
  //commandline=get_command(commandline,4,e,-1,-1);
  commandline += "<cr>";
  return commandline;
}

int readNumberFromSerial() {
  String numberString = Serial.readStringUntil('\n'); // 读取串口输入直到换行符为止
  
  int number = numberString.toInt();  // 将字符串转换为整数
  // 限制数字的范围在0到2500之间
  Serial.print(number);
  return constrain(number, 0, 2500);
}

char inByte=0;
int input;
String command="";

int fsr0_reading;
int fsr1_reading;


void loop() {
  command="";
  if (Serial.available() > 0) {
    // get incoming byte:
    // inByte = Serial.read();
    // flag=MK_Port(inByte);
    input=readNumberFromSerial();
    //Serial.println("input");
    //Serial.println(input);
  }
  
  if (input==1){
    command=summon_command(1500,1300,1400,600,-1);
    Serial1.println(command);
    input=0; 
  }

  if (input==2){
    command=summon_command(1500,1300,1400,600,-1);
    Serial1.println(command);
    input=0; 
  }
  
  if (input==3){
    command=summon_command(1500,2000,200,600,500);
    Serial1.println(command);

    for(int i=0;i<=9; i++) {
       // read from fsr
      if (angle>=5) {
        angle -= 5; 
        if (angle <= 0) {
          angle = 180;
          delay(5000);
        }
        if (angle > 180) angle = 180;
        Serial.println("angle");
        Serial.println(angle);
        myservo.write(angle);  
      }
  
      fsr0_reading = getPressure(A0,"A0");
      fsr1_reading = getPressure(A1,"A1");
  
      if ((fsr0_reading>0) || (fsr1_reading>0)) {
        read_rgb();
      }

      delay(1);
    }

    input = 0;
    command=summon_command(1500,2000,200,600,2500);
  }

}
