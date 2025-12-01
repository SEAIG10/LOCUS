import { EventEmitter } from 'events';

// 전역 이벤트 버스
export const eventBus = new EventEmitter();

// 이벤트 이름 정의
export const EVENTS = {
  NEW_POLLUTION_PREDICTION: 'NEW_POLLUTION_PREDICTION', // MQTT -> 오염 예측
  NEW_SENSOR_EVENT: 'NEW_SENSOR_EVENT',               // MQTT -> 청소/센서 알림
  NEW_ROBOT_LOCATION: 'NEW_ROBOT_LOCATION',           // Mobile App -> 로봇 위치
};