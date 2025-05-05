#define GREEN_LED 13  
#define RED_LED 12  

String incomingData = "";  

void setup() {
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  Serial.begin(9600);  
}

void loop() {
  if (Serial.available()) {
    incomingData = Serial.readStringUntil('\n');
    incomingData.trim();

    if (incomingData == "ON") {
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(RED_LED, LOW);
      blinkLED(GREEN_LED);  // Blink green LED to indicate "ON" received
    }
    else if (incomingData == "OFF") {
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(RED_LED, HIGH);
      blinkLED(RED_LED);  // Blink red LED to indicate "OFF" received
    }
  }
}

void blinkLED(int ledPin) {
  digitalWrite(ledPin, HIGH);
  delay(500);  // LED stays on for 500ms
  digitalWrite(ledPin, LOW);
  delay(500);  // LED stays off for 500ms
}