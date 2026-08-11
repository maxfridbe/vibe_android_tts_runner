plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Injected by build.sh; null only for builds outside the build system,
// which then fall back to the default debug signing.
val sharedKeystore = findProperty("appKeystoreFile") as String?

android {
    namespace = "com.techhurts.ttsrunner"
    compileSdk = 34
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.techhurts.ttsrunner"
        minSdk = 29        // vendor OpenCL dlopen path + typed foreground services
        targetSdk = 34
        // Injected by build.sh from git (commit count / git describe);
        // the defaults only apply to builds outside the build system.
        versionCode = (findProperty("appVersionCode") as String?)?.toInt() ?: 1
        versionName = findProperty("appVersionName") as String? ?: "0.0.0-dev"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    // native code is useless unoptimized; keep Release even for assembleDebug
                    "-DCMAKE_BUILD_TYPE=Release",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    if (sharedKeystore != null) {
        signingConfigs.create("shared") {
            storeFile = file(sharedKeystore)
            storePassword = findProperty("appKeystorePassword") as String?
            keyAlias = findProperty("appKeyAlias") as String? ?: "androidbase"
            keyPassword = findProperty("appKeyPassword") as String?
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (sharedKeystore != null) {
                signingConfig = signingConfigs.getByName("shared")
            }
        }
        debug {
            // Same key for debug so signatures never change between machines
            // and `adb install -r` upgrades always work.
            if (sharedKeystore != null) {
                signingConfig = signingConfigs.getByName("shared")
            }
        }
    }

    buildFeatures {
        buildConfig = true   // BuildConfig.VERSION_NAME shown in the UI + debug report
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    implementation("org.jsoup:jsoup:1.17.2") // article extraction for shared URLs
    implementation("net.dankito.readability4j:readability4j:1.0.8") // Mozilla Readability.js port
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0") // M3 tabs/nav/cards
    // Supertonic 3 runs as ONNX graphs; this ships the ORT native libs plus the
    // NNAPI execution provider used for the GPU/NPU attempt
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
