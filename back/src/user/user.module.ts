import { Module } from '@nestjs/common';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import { UserRepository } from './repositories/user.repository';
import { HealthInfoRepository } from './repositories/health-info.repository';

@Module({
  controllers: [UserController],
  providers: [UserService, UserRepository, HealthInfoRepository]
})
export class UserModule {}
