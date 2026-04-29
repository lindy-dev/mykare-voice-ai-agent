import { NextResponse } from 'next/server';

// NOTE: you are expected to define the following environment variables in `.env.local`:
const USERNAME = process.env.AUTH_USERNAME;
const PASSWORD = process.env.AUTH_PASSWORD;

export async function POST(req: Request) {
  try {
    const { username, password } = await req.json();

    if (username === USERNAME && password === PASSWORD) {
      return NextResponse.json({ success: true });
    }

    return NextResponse.json({ error: 'Invalid username or password' }, { status: 401 });
  } catch {
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
  }
}
