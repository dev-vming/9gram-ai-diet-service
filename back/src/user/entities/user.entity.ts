import { IsBoolean, IsDate, IsEmail, IsEnum, IsNotEmpty, IsOptional, IsString, IsUUID } from 'class-validator';
import { GoogleLoginDto } from '../../auth/dto/googleLoginDto';
import { Column, CreateDateColumn, DeleteDateColumn, Entity, JoinColumn, OneToMany, OneToOne, PrimaryColumn, Timestamp, UpdateDateColumn } from 'typeorm';
import { v4 as uuidv4 } from 'uuid';
import { LocalSignupDto } from '../../auth/dto/localSignupDto';
import * as bcrypt from 'bcrypt';
import { HealthInfo } from './health-info.entity';
import { UpdateUserDto } from '../dto/UpdateUser.dto';
import { SaveHealthInfoDto } from '../dto/SaveHealthInfo.dto'
import { Gender } from '../utils/user.enums';

@Entity('user')
export class User {
  @PrimaryColumn({type:'uuid', nullable: false, unique: true})
  @IsUUID()
  @IsNotEmpty()
  userId: string;

  @Column({type:'varchar',length:50, nullable: false, unique: true})
  @IsEmail()
  @IsNotEmpty()
  email: string;

  @Column({type:'varchar',length:50, nullable: true, unique: true})
  @IsOptional()
  providerId: string;

  @Column({type:'varchar',length:200, nullable: true})
  @IsString()
  @IsOptional()
  password: string;

  @Column({type:'varchar',length:50, nullable: false, unique: true})
  @IsString()
  @IsNotEmpty()
  username: string;

  @Column({type:'date', nullable: true})
  @IsDate()
  @IsOptional()
  birthDay: Date;

  @Column({ type: 'varchar', nullable: true})
  @IsEnum(Gender)
  @IsOptional()
  gender: string;

  @Column({ type: 'varchar', length: 50, nullable: true })
  @IsString()
  @IsOptional()
  profileImage: string;

  @Column({ type: 'boolean', default: false })
  @IsBoolean()
  @IsOptional()
  membership: boolean;

  @Column({ type: 'timestamp with time zone', default: () => 'CURRENT_TIMESTAMP' })
  @CreateDateColumn()
  createdDate: Timestamp;

  @Column({ type: 'timestamp with time zone', default: () => 'CURRENT_TIMESTAMP'})
  @UpdateDateColumn()
  updatedDate: Timestamp;

  // @Column({ type: 'timestamp with time zone', nullable: true })
  @DeleteDateColumn()
  deletedDate: Timestamp;

  // @OneToMany((type) => HealthInfo, healthInfo => healthInfo.user, { cascade: true })
  // healthInfos: HealthInfo[];

  @Column({type:'uuid', nullable: true})
  recentHealthInfoId: string;

  public mapGoogleLoginDto(googleLoginDto: GoogleLoginDto):User {
    const {providerId, email, displayName} = googleLoginDto;
    const user = new User();
    user.userId = uuidv4();
    user.providerId = providerId;
    user.email = email;
    user.username = displayName;
    return user;
  }

  public mapLocalSignupDto(localSignupDto: LocalSignupDto):User {
    const {email, password, username} = localSignupDto;
    const user = new User();
    user.userId = uuidv4();
    user.username = username;
    user.email = email;
    const salt = bcrypt.genSaltSync(10);
    const hashedPassword = bcrypt.hashSync(password, salt);
    user.password = hashedPassword;
    return user;
  }

  public mapUpdateUserDto(dto: UpdateUserDto){
    const user = new User();
    user.username = dto.username;
    // user.password = dto.password;
    user.birthDay = dto.birthDay;
    user.gender = dto.gender;
    user.profileImage = dto.profileImage;
    user.membership = dto.membership;
    return user;
  }
}