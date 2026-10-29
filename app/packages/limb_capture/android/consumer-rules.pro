# ARCore, MediaPipe and protobuf-lite are called from native code by class/method name: never rename or strip them.
-keep class com.google.ar.** { *; }
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keep class com.spentys.limb_capture.** { *; }
-dontwarn com.google.ar.**
-dontwarn com.google.mediapipe.**
-dontwarn com.google.protobuf.**
# MediaPipe logs through Flogger, which finds its caller by class name on the stack (breaks when renamed).
-keep class com.google.common.flogger.** { *; }
-dontwarn com.google.common.flogger.**
