#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);       // Signal pin connected to D9
  myServo.write(90);       // Neutral position
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    if (command == 'B') {
      myServo.write(0);    // Move to 0°
      delay(100);
      myServo.write(90);   // Return to 90°
      delay(100);
    }
  }
}