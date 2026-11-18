import { render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("@react-three/fiber", () => ({
  Canvas: ({ children }) => <div>{children}</div>,
  useLoader: () => null,
}));

jest.mock("@react-three/drei", () => ({
  OrbitControls: () => null,
  Line: () => null,
}));

jest.mock("three", () => ({
  Box3: function Box3() {},
  Vector3: function Vector3() {},
}));

jest.mock("three-stdlib", () => ({
  PLYLoader: function PLYLoader() {},
  OBJLoader: function OBJLoader() {},
  GLTFLoader: function GLTFLoader() {},
}));

test("shows login before an account is stored", () => {
  localStorage.clear();
  render(<App />);
  expect(screen.getByRole("button", { name: /^log in$/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /sign up/i })).toBeInTheDocument();
});
