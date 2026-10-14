using System;
using System.IO;
using System.Runtime.InteropServices;
var module=NativeLibrary.Load("ucrtbase.dll");var function=NativeLibrary.GetExport(module,"expf");var expf=Marshal.GetDelegateForFunctionPointer<NativeExpf>(function);
var source=File.ReadAllBytes(args[0]);var output=new byte[source.Length];
for(int i=0;i<source.Length;i+=4)BitConverter.GetBytes(expf(BitConverter.ToSingle(source,i))).CopyTo(output,i);
File.WriteAllBytes(args[1],output);Console.WriteLine($"NativeCRT exponential oracle {source.Length/4} arguments");
[UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate float NativeExpf(float x);
