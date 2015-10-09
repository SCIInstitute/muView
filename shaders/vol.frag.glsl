uniform sampler3D sf_texture;

varying vec3 myTexCoord;


void main (void){

    gl_FragColor = texture3D( sf_texture, myTexCoord );
    //gl_FragColor = vec4(t,t,t,0.0);

    //if(t > 0 && t < 0.1) gl_FragColor = vec4(1,0,0,0.2);

    //if( t <= 0.5f ) gl_FragColor = lerp( vec4(1,0,0,0.2), vec4(0,0,1,0.2), (t-0.0f)*2.0f );
    //if( t >= 0.5f ) gl_FragColor = lerp( vec4(0,0,1,0.2), vec4(1,1,1,0.2), (t-0.5f)*2.0f );

    //gl_FragColor = vec4(1,0,0,0.01);

    //if( t <= 0.001f ) gl_FragColor = vec4(0,0,0,0);
    if(myTexCoord.x < 0 || myTexCoord.x > 1) gl_FragColor = vec4(0,0,0,0);
    if(myTexCoord.y < 0 || myTexCoord.y > 1) gl_FragColor = vec4(0,0,0,0);
    if(myTexCoord.z < 0 || myTexCoord.z > 1) gl_FragColor = vec4(0,0,0,0);

}

