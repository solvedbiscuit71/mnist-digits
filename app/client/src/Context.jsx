import { createContext } from "react";

export const APIContext = createContext(import.meta.env.DEV ? 'http://127.0.0.1:5000/api' : '/api')