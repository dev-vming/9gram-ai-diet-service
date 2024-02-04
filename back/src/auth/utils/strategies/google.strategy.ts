import { AuthService } from '../../auth.service';
import { User } from '../../../user/entities/user.entity';
import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Profile, Strategy } from 'passport-google-oauth20';

@Injectable()
// ❶ PassportStrategy(Strategy) 상속
export class GoogleStrategy extends PassportStrategy(Strategy) {
  constructor(private authService: AuthService) {  // ❷ 생성자

// ❸ 부모 클래스의 생성자를 호출
    super({
      clientID: process.env.GOOGLE_CLIENT_ID,            // 클라이언트 ID
      clientSecret: process.env.GOOGLE_CLIENT_PW,    // 시크릿
      callbackURL: process.env.GOOGLE_CALLBACK_URL,  // 콜백 URL
      scope: ['email', 'profile'],                       // scope
    });
  }

// ❹ OAuth 인증이 끝나고 콜백으로 실행되는 메서드
  async validate(accessToken: string, refreshToken: string, profile: Profile) {
    
    // 등록된 구글계정인지 확인, 없으면 등록
    const user: User = await this.authService.validateGoogleOauthUser({
      providerId: profile.id,
      email: profile.emails[0].value,
      displayName: profile.displayName,
    });

    return user || null ;
  }
}