#include <Servo.h>

Servo __myservo16;
Servo __myservo11;
Servo __myservo9;
Servo __myservo8;
Servo __myservo7;

String str;

unsigned int servo16_value, servo11_value, servo9_value, servo8_value, servo7_value;

String SA[2];
int A[2];

void setup(void)
{
  Serial.begin(9600);
  __myservo16.attach(16);
  __myservo11.attach(11);
  __myservo9.attach(9);
  __myservo8.attach(8);
  __myservo7.attach(7);

  servo16_value = 90;
  servo11_value = 45;
  servo9_value = 0;
  servo8_value = 135;
  servo7_value = 135;

  __myservo16.write(servo16_value);
  __myservo11.write(servo11_value);
  __myservo9.write(servo9_value);
  __myservo8.write(servo8_value);
  __myservo7.write(servo7_value);


}

void loop(void)
{
  if (Serial.available()) {
    // 讀取傳入的字串直到"\n"結尾
    str = Serial.readStringUntil('\n');

    String_to_Int(str, 2);
    Serial.println(SA[0]);
    Serial.println(A[1]);
  }

  if (SA[0] == "16") {
    int PRE_angle = __myservo16.read();
    if (A[1] >= PRE_angle) {
      for (int angle=PRE_angle;angle<=A[1];angle++){
         __myservo16.write(angle);
         delay(20);
      }
      Serial.println("Servo_OK");
    }
    if (A[1] < PRE_angle) {
      for (int angle=PRE_angle;angle>=A[1];angle--){
         __myservo16.write(angle);
         delay(20);
      }
      Serial.println("Servo_OK");
    }    
  }
  if (SA[0] == "11") {
    int PRE_angle = __myservo11.read();
    if (A[1] >= PRE_angle) {
      for (int angle=PRE_angle;angle<=A[1];angle++){
         __myservo11.write(angle);
         delay(20);
      }
    }
    if (A[1] < PRE_angle) {
      for (int angle=PRE_angle;angle>=A[1];angle--){
         __myservo11.write(angle);
         delay(20);
      }
    }    
  }
  if (SA[0] == "9") {
    int PRE_angle = 180 - __myservo9.read();
    if (A[1] >= PRE_angle) {
      for (int angle=PRE_angle;angle<=A[1];angle++){
         __myservo9.write(180-angle);
         delay(20);
      }
    }
    if (A[1] < PRE_angle) {
      for (int angle=PRE_angle;angle>=A[1];angle--){
         __myservo9.write(180-angle);
         delay(20);
      }
    }    
  }
  if (SA[0] == "8") {
    int PRE_angle = __myservo8.read();
    if (A[1] >= PRE_angle) {
      for (int angle=PRE_angle;angle<=A[1];angle++){
         __myservo8.write(angle);
         delay(20);
      }
    }
    if (A[1] < PRE_angle) {
      for (int angle=PRE_angle;angle>=A[1];angle--){
         __myservo8.write(angle);
         delay(20);
      }
    }    
  }
  if (SA[0] == "7") {
    __myservo7.write(A[1]);
  }
  
}

void String_to_Int(String temp, int count)
{
  int index;

  index = temp.indexOf(',');
  SA[0] = temp.substring(0, index);

  for (int i = 1; i < count; i++) {
    temp = temp.substring(index + 1, temp.length());
    index = temp.indexOf(',');
    SA[i] = temp.substring(0, index);
    A[i] = SA[i].toInt();
  }
}
