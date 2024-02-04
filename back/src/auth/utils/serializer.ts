import { AuthService } from '../auth.service';
import { Inject, Injectable } from '@nestjs/common';
import { PassportSerializer } from '@nestjs/passport';
import { User } from '../../user/entities/user.entity';

@Injectable()
export class SessionSerializer extends PassportSerializer {
  constructor(@Inject(AuthService) private readonly authService: AuthService) {
    super();
  }
  
  // 세션에 사용자 정보를 저장
  serializeUser(user: User, done: Function) {
    // console.log('SerializeUser');
    done(null, user.userId);
  }
  
  // 세션에서 사용자 정보를 읽어서 req.user에 저장
  async deserializeUser(payload: any, done: Function) {
    // console.log('DeserializeUser');
    const user = await this.authService.findOneByUserId(payload);  // payload는 serializeUser에서 done(null, user.userId)로 넘겨준 값
    return user ? done(null, user) : done(null, null);
  }
}