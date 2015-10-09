varying vec3 myTexCoord;

void main(void){
	myTexCoord = gl_MultiTexCoord0.xyz;
	gl_Position = ftransform();
}

