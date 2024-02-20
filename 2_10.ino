#include <Servo.h> 
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
  // fsr setup
  myservo.attach(8); 
  angle=180;
  myservo.write(angle);

  // rgb setup
  Serial.begin(9600);
  pinMode(s0,OUTPUT);
  pinMode(s1,OUTPUT);
  pinMode(s2,OUTPUT);
  pinMode(s3,OUTPUT);
  FlexiTimer2::set(50,isr);
  FlexiTimer2::start();
  
  // delay(5000);  // wait 5 seconds for placing object
}


void getPressure(int analogInput, String str) {
  int fsrReading = analogRead(analogInput); 
  //Serial.print(str); Serial.print(","); 
  Serial.println(fsrReading);
  delay(10);
}


void isr(){  //the timer 2, 10ms interrupt overflow again. Internal overflow interrupt executive function
  // isr: interrupt service routine
    TCNT2=100;
  
    countR= pulseIn(sensorOut, LOW);
    // 测量的是信号在一个LOW期间的持续时间，具体来说，测量的是以下两点间的时间间隔
    // 信号从HIGH变为LOW的那一刻
    // 信号从LOW回到HIGH的那一刻
    // 在强光照射下，光传感器的输出信号可能会一直保持在HIGH状态,从而没有下降到LOW的那个下降，因此始终检测不到这个LOW的pulse，因此返回不了
    // 原因：当光传感器暴露在非常强的光线下时（比如直射的阳光或强烈的人造光源），它可能会被光“饱和”。这意味着传感器被光线过度刺激，以至于无法正确地响应光线强度的变化。在这种“饱和”状态下，传感器可能会持续输出HIGH信号，因为它检测到的光强远远超出了其正常工作范围。
    // 就像当你直视强光时眼睛会一时看不清其他东西一样，强光下光传感器也可能“看不清”光强的真实变化，从而持续输出一个高电平的信号
    //Serial.print("red=");
    Serial.println(countR/10);
    delay(50);
    digitalWrite(s2,HIGH);
    digitalWrite(s3,HIGH);
    
    countG= pulseIn(sensorOut, LOW);
    //Serial.print("green=");
    Serial.println(countG/10);
    delay(50);
    digitalWrite(s2,LOW);
    digitalWrite(s3,HIGH);
    
    countB= pulseIn(sensorOut, LOW);
    //Serial.print("blue=");
    Serial.println(countB/10);
    delay(50);
    Serial.println("\n");
    digitalWrite(s2,LOW);
    digitalWrite(s3,LOW);
}


void TCS() {
  flag=0;
  digitalWrite(s1,HIGH);
  digitalWrite(s0,HIGH);
  digitalWrite(s2,LOW);
  digitalWrite(s3,LOW);
  attachInterrupt(0, ISR_INTO, CHANGE);
  timer0_init();
}

void ISR_INTO()
{
  counter++;
}

void timer0_init(void)
{
  TCCR2A=0x00;
  TCCR2B=0x07; //the clock frequency source 1024 points
  TCNT2= 100; //10 ms overflow again
  TIMSK2 = 0x01; //allow interrupt
}

int i=0;


void loop()
{
  TCS();
  //while(1);
  if (angle>=5) {
    angle -= 5; 
    if (angle <= 0) {
      angle = 180;
      delay(5000);
    }
    if (angle > 180) angle = 180;
    //Serial.println("angle");
    Serial.println(angle);
    myservo.write(angle);  
    delay(100);
  }
  getPressure(A0,"A0");
  getPressure(A1,"A1");
  delay(100);
}
