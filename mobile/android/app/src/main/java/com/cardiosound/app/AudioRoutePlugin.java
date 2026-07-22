package com.cardiosound.app;

import android.content.Context;
import android.media.AudioManager;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "AudioRoute")
public class AudioRoutePlugin extends Plugin {
    
    @PluginMethod
    public void setSpeaker(PluginCall call) {
        boolean enable = call.getBoolean("enable", true);
        
        try {
            AudioManager audioManager = (AudioManager) getContext().getSystemService(Context.AUDIO_SERVICE);
            if (audioManager != null) {
                // Configurar el modo para permitir redirigir el audio durante grabación/reproducción
                audioManager.setMode(AudioManager.MODE_IN_COMMUNICATION);
                audioManager.setSpeakerphoneOn(enable);
                call.resolve();
            } else {
                call.reject("AudioManager not available");
            }
        } catch (Exception e) {
            call.reject("Error setting speaker: " + e.getMessage());
        }
    }
}
