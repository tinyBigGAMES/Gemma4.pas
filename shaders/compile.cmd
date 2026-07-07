@echo off
setlocal

echo ============================================================
echo Gemma4 Shader Build Pipeline
echo ============================================================
echo.

rem Build all shader variants (GLSL -> SPIR-V)
python build_shaders.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo SHADER COMPILATION FAILED
    exit /b 1
)

rem Compile .rc -> .res for Delphi linking
echo.
echo Compiling shader resources...
"C:\Program Files (x86)\Embarcadero\Studio\23.0\bin\brcc32.exe" Gemma4.Shaders.rc -fo..\src\Gemma4.Shaders.res
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: Resource compilation
    exit /b 1
)

echo.
echo ============================================================
echo Build complete. Output: ..\src\Gemma4.Shaders.res
echo ============================================================
