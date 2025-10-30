import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface User {
  id: string | { timestamp: number; date: string };
  email: string;
  roles: string[];
}

interface AuthState {
  token: string | null;
  user: User | null;
  isAdmin: boolean;
}

const initialState: AuthState = {
  token: localStorage.getItem('agroml_token'),
  user: null,
  isAdmin: false,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (state, action: PayloadAction<{ token: string; user: User; isAdmin: boolean }>) => {
      state.token = action.payload.token;
      state.user = action.payload.user;
      state.isAdmin = action.payload.isAdmin;
      localStorage.setItem('agroml_token', action.payload.token);
    },
    logout: (state) => {
      state.token = null;
      state.user = null;
      state.isAdmin = false;
      localStorage.removeItem('agroml_token');
    },
  },
});

export const { setCredentials, logout } = authSlice.actions;
export default authSlice.reducer;
