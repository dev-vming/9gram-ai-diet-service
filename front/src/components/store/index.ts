import { configureStore } from '@reduxjs/toolkit';
import userLoginReducer from './userLoginRouter';

const store = configureStore({
  reducer: {
    user: userLoginReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
