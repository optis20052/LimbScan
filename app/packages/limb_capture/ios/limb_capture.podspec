#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint limb_capture.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'limb_capture'
  s.version          = '0.0.1'
  s.summary          = 'Native AR capture (ARCore / ARKit): sharp frames + metric pose + intrinsics'
  s.description      = <<-DESC
Native AR capture (ARCore / ARKit): sharp frames + metric pose + intrinsics
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'limb_capture/Sources/limb_capture/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '15.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'limb_capture_privacy' => ['limb_capture/Sources/limb_capture/PrivacyInfo.xcprivacy']}
end
