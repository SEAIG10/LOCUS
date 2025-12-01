/**
 * App.tsx
 * LOCUS Tracker (Direct to Backend)
 */

import React, { useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { useExpoARTracking } from './hooks/useExpoARTracking';
import { useRestApiSender } from './hooks/useRestApiSender';

// ⚠️ 백엔드 서버의 /api/log/record 주소로 정확히 입력하세요
// ngrok을 쓴다면: https://xxxx.ngrok-free.app/api/log/record
// 1. 기본 도메인 (Ngrok 주소가 바뀌면 여기만 변경!)

const SERVER_HOST = 'https://fad6f8263bcc.ngrok-free.app';
// 2. API 경로 (백엔드 라우터)

const API_PATH = '/api/log/record';
// 3. 최종 URL 자동 조합

const BACKEND_API_URL = `${SERVER_HOST}${API_PATH}`;

export default function App() {
  // 1. AR 위치 추적 훅
  const {
    position,
    trackingState,
    accuracy,
    yaw,
    isTracking,
    error: arError,
    startTracking,
    stopTracking,
    resetOrigin,
  } = useExpoARTracking({
    autoStart: true,
    onError: (err) => console.error('ARKit 오류:', err),
  });

  // 2. HTTP API 전송 훅 (WebSocket 대체)
  const {
    isSending,
    error: apiError,
    sendPosition,
    clientId,
  } = useRestApiSender({
    serverUrl: BACKEND_API_URL,
    interval: 200, // 0.2초마다 전송 (너무 빠르면 서버 과부하)
  });

  // 3. 위치 업데이트 시 서버 전송
  useEffect(() => {
    // 위치가 있고, 추적 중일 때만 전송
    if (position && isTracking) {
      sendPosition(position, accuracy);
    }
  }, [position, isTracking, accuracy, sendPosition]);

  // UI 헬퍼: 추적 상태 색상
  const getTrackingColor = () => {
    switch (trackingState) {
      case 'normal': return '#4ecca3';
      case 'initializing': return '#ffa502';
      case 'not_available': return '#ff6348';
      default: return '#95afc0';
    }
  };

  // UI 헬퍼: 추적 상태 텍스트
  const getTrackingText = () => {
    switch (trackingState) {
      case 'normal': return '🟢 정상 추적';
      case 'initializing': return '🟡 초기화 중';
      case 'not_available': return '🔴 추적 불가';
      case 'excessive_motion': return '⚠️ 움직임 과다';
      case 'insufficient_features': return '⚠️ 특징점 부족';
      default: return '⚪ 대기 중';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0f0f1e" />
      
      <ScrollView style={styles.scrollView}>
        
        {/* 헤더 */}
        <View style={styles.header}>
          <Text style={styles.title}>LOCUS Tracker</Text>
          <Text style={styles.subtitle}>Direct Backend Mode</Text>
          <Text style={styles.version}>Client ID: #{clientId}</Text>
        </View>

        {/* ARKit 상태 */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>ARKit 상태</Text>
          <View style={styles.statusRow}>
            <Text style={styles.label}>추적 상태</Text>
            <Text style={[styles.value, { color: getTrackingColor() }]}>
              {getTrackingText()}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.label}>정확도</Text>
            <Text style={styles.value}>
              ±{(accuracy * 100).toFixed(1)} cm
            </Text>
          </View>
        </View>

        {/* 현재 위치 */}
        {position && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>현재 위치</Text>
            <View style={styles.coordinateRow}>
              <Text style={styles.coordinateLabel}>X</Text>
              <Text style={styles.coordinateValue}>
                {position.x.toFixed(3)} m
              </Text>
            </View>
            <View style={styles.coordinateRow}>
              <Text style={styles.coordinateLabel}>Y</Text>
              <Text style={styles.coordinateValue}>
                {position.y.toFixed(3)} m
              </Text>
            </View>
            <View style={styles.coordinateRow}>
              <Text style={styles.coordinateLabel}>Z</Text>
              <Text style={styles.coordinateValue}>
                {position.z.toFixed(3)} m
              </Text>
            </View>
            <View style={styles.coordinateRow}>
              <Text style={styles.coordinateLabel}>방향</Text>
              <Text style={styles.coordinateValue}>
                {yaw.toFixed(1)}°
              </Text>
            </View>
          </View>
        )}

        {/* API 전송 상태 (구 WebSocket 패널 대체) */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>백엔드 전송</Text>
          <View style={styles.statusRow}>
            <Text style={styles.label}>상태</Text>
            <Text style={[styles.value, { color: apiError ? '#ff6348' : '#4ecca3' }]}>
              {apiError 
                ? '🔴 전송 실패' 
                : (isTracking ? '🟢 전송 중 (HTTP)' : '⚪ 대기 중')}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.label}>API URL</Text>
            <Text style={[styles.value, { fontSize: 10, maxWidth: 200 }]} numberOfLines={1}>
              {BACKEND_API_URL}
            </Text>
          </View>
        </View>

        {/* 오류 표시 */}
        {(arError || apiError) && (
          <View style={[styles.card, styles.errorCard]}>
            <Text style={styles.errorTitle}>⚠️ 오류 발생</Text>
            {arError && <Text style={styles.errorText}>ARKit: {arError}</Text>}
            {apiError && <Text style={styles.errorText}>API: {apiError}</Text>}
          </View>
        )}

        {/* 컨트롤 버튼 */}
        <View style={styles.controls}>
          <TouchableOpacity
            style={[styles.button, isTracking ? styles.buttonDanger : styles.buttonPrimary]}
            onPress={isTracking ? stopTracking : startTracking}>
            <Text style={styles.buttonText}>
              {isTracking ? '⏸ 추적 중지' : '▶️ 추적 시작'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.buttonSecondary]}
            onPress={resetOrigin}>
            <Text style={styles.buttonText}>📍 원점 재설정</Text>
          </TouchableOpacity>
        </View>

        {/* 안내 */}
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>💡 다이렉트 모드 안내</Text>
          <Text style={styles.infoText}>
            • 중계 서버 없이 백엔드로 직접 전송합니다.
          </Text>
          <Text style={styles.infoText}>
            • 데이터 절약을 위해 0.2초마다 전송됩니다.
          </Text>
          <Text style={styles.infoText}>
            • Client ID #{clientId}로 로그가 저장됩니다.
          </Text>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f1e',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    padding: 24,
    paddingTop: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#95afc0',
    marginBottom: 4,
  },
  version: {
    fontSize: 14,
    color: '#4ecca3',
    fontWeight: '600',
  },
  card: {
    backgroundColor: '#1a1a2e',
    margin: 16,
    marginTop: 0,
    padding: 20,
    borderRadius: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 16,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    color: '#95afc0',
  },
  value: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  coordinateRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#2d2d44',
  },
  coordinateLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#4ecca3',
    width: 50,
  },
  coordinateValue: {
    fontSize: 18,
    fontWeight: '500',
    color: '#ffffff',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  errorCard: {
    backgroundColor: '#3d1f1f',
    borderWidth: 1,
    borderColor: '#ff6348',
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ff6348',
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#ffb8b0',
    marginBottom: 4,
  },
  controls: {
    padding: 16,
    gap: 12,
  },
  button: {
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonPrimary: {
    backgroundColor: '#4ecca3',
  },
  buttonDanger: {
    backgroundColor: '#ff6348',
  },
  buttonSecondary: {
    backgroundColor: '#535c68',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  infoCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 32,
    padding: 20,
    backgroundColor: '#1e2a3a',
    borderRadius: 16,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffa502',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#95afc0',
    marginBottom: 8,
    lineHeight: 20,
  },
});